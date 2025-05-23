import torch
import numpy as np
from typing import Any, Dict, Hashable, Mapping, Tuple
from models.CLIP import CLIPModel
from loralib.utils import apply_lora, mark_only_lora_as_trainable
from monai import transforms as monai_transforms
from monai.config.type_definitions import NdarrayOrTensor
import torchvision
from monai.transforms import MapTransform, Transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(args):
    """
    Initialize the model based on the provided arguments.
    Args:
        args: Arguments loaded from the saved model
    Returns:
        model: CLIP model
    """
    model = CLIPModel(args)

    if "ft" in args.tuning:
        print("Full tuning")
    elif args.tuning == "pt":
        print("Probe tuning")
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if (
                "image_projection" in name
                or "semantic_projection" in name
                or "classifier_image" in name
                or "classifier_text" in name
                or "logit_scale" in name
            ):
                param.requires_grad = True

    elif args.tuning == "lora":
        print("LoRA tuning")
        args.backbone = args.model.split("_")[1]
        list_lora_layers = apply_lora(args, model)
        mark_only_lora_as_trainable(model)

        for name, param in model.named_parameters():
            if (
                "image_projection" in name
                or "semantic_projection" in name
                or "classifier_image" in name
                or "classifier_text" in name
                or "logit_scale" in name
            ):
                param.requires_grad = True

    model = model.to(device)
    return model


'''
The follwing function is adapted from foundation cancer image biomarker model code
https://github.com/AIM-Harvard/foundation-cancer-image-biomarker
'''

def get_transforms_raw(spatial_size=(50, 50, 50), precropped=False, jitter=None):
    return monai_transforms.Compose(
        [
            monai_transforms.LoadImaged(
                keys=["image_path"], image_only=True, reader="ITKReader"
            ),
            monai_transforms.EnsureChannelFirstd(keys=["image_path"]),
            monai_transforms.Spacingd(
                keys=["image_path"],
                pixdim=1,
                padding_mode="zeros",
                mode="linear",
                align_corners=True,
                diagonal=True,
            ),
            monai_transforms.Orientationd(keys=["image_path"], axcodes="LPS"),
            SeedBasedPatchCropd(
                keys=["image_path"],
                roi_size=spatial_size[::-1],
                coord_orientation="LPS",
                global_coordinates=True,
                jitter=jitter,
            ),
            monai_transforms.SelectItemsd(keys=["image_path"]),
            monai_transforms.Transposed(keys=["image_path"], indices=(0, 3, 2, 1)),
            monai_transforms.SpatialPadd(
                keys=["image_path"], spatial_size=spatial_size
            ),
            torchvision.transforms.Lambda(lambda x: x["image_path"].as_tensor()),
        ]
    )


class SeedBasedPatchCropd(MapTransform):
    """
    A class representing a seed-based patch crop transformation.

    Inherits from MapTransform.

    Attributes:
        keys (list): List of keys for images in the input data dictionary.
        roi_size (tuple): Tuple indicating the size of the region of interest (ROI).
        allow_missing_keys (bool): If True, do not raise an error if some keys in the input data dictionary are missing.
        coord_orientation (str): Coordinate system (RAS or LPS) of input coordinates.
        global_coordinates (bool): If True, coordinates are in global coordinates; otherwise, local coordinates.
    """

    def __init__(
        self,
        keys,
        roi_size,
        allow_missing_keys=False,
        coord_orientation="RAS",
        global_coordinates=True,
        jitter=None,
    ) -> None:
        """
        Initialize SeedBasedPatchCropd class.

        Args:
            keys (List): List of keys for images in the input data dictionary.
            roi_size (Tuple): Tuple indicating the size of the region of interest (ROI).
            allow_missing_keys (bool): If True, do not raise an error if some keys in the input data dictionary are missing.
            coord_orientation (str): Coordinate system (RAS or LPS) of input coordinates.
            global_coordinates (bool): If True, coordinates are in global coordinates; otherwise, local coordinates.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.coord_orientation = coord_orientation
        self.global_coordinates = global_coordinates
        self.cropper = SeedBasedPatchCrop(roi_size=roi_size)
        self.jitter = jitter

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Apply transformation to given data.

        Args:
            data (dict): Dictionary with image keys and required center coordinates.

        Returns:
            dict: Dictionary with cropped patches for each key in the input data dictionary.
        """
        d = dict(data)

        assert "coordX" in d.keys(), "coordX not found in data"
        assert "coordY" in d.keys(), "coordY not found in data"
        assert "coordZ" in d.keys(), "coordZ not found in data"

        # Jitter the seed coordinates
        if self.jitter is not None:
            d["coordX"] += np.random.uniform(-self.jitter, self.jitter)
            d["coordY"] += np.random.uniform(-self.jitter, self.jitter)
            d["coordZ"] += np.random.uniform(-self.jitter, self.jitter)

        # Convert coordinates to RAS orientation to match image orientation
        if self.coord_orientation == "RAS":
            center = (d["coordX"], d["coordY"], d["coordZ"])
        elif self.coord_orientation == "LPS":
            center = (-d["coordX"], -d["coordY"], d["coordZ"])

        for key in self.key_iterator(d):
            d[key] = self.cropper(
                d[key], center=center, global_coordinates=self.global_coordinates
            )
        return d


class SeedBasedPatchCrop(Transform):
    """
    A class representing a seed-based patch crop transformation.

    Attributes:
        roi_size: Tuple indicating the size of the region of interest (ROI).

    Methods:
        __call__: Crop a patch from the input image centered around the seed coordinate.

    Args:
        roi_size: Tuple indicating the size of the region of interest (ROI).

    Returns:
        NdarrayOrTensor: Cropped patch of shape (C, Ph, Pw, Pd), where (Ph, Pw, Pd) is the patch size.

    Raises:
        AssertionError: If the input image has dimensions other than (C, H, W, D)
        AssertionError: If the coordinates are invalid (e.g., min_h >= max_h)
    """

    def __init__(self, roi_size) -> None:
        """
        Initialize SeedBasedPatchCrop class.

        Args:
            roi_size (tuple): Tuple indicating the size of the region of interest (ROI).
        """
        super().__init__()
        self.roi_size = roi_size

    def __call__(
        self, img: NdarrayOrTensor, center: tuple, global_coordinates=False
    ) -> NdarrayOrTensor:
        """
        Crop a patch from the input image centered around the seed coordinate.

        Args:
            img (NdarrayOrTensor): Image to crop, with dimensions (C, H, W, D). C is the number of channels.
            center (tuple): Seed coordinate around which to crop the patch (X, Y, Z).
            global_coordinates (bool): If True, seed coordinate is in global space; otherwise, local space.

        Returns:
            NdarrayOrTensor: Cropped patch of shape (C, Ph, Pw, Pd), where (Ph, Pw, Pd) is the patch size.
        """
        assert len(img.shape) == 4, "Input image must have dimensions: (C, H, W, D)"
        C, H, W, D = img.shape
        Ph, Pw, Pd = self.roi_size

        # If global coordinates, convert to local coordinates
        if global_coordinates:
            center = np.linalg.inv(np.array(img.affine)) @ np.array(center + (1,))
            center = [int(x) for x in center[:3]]

        # Calculate and clamp the ranges for cropping
        min_h, max_h = max(center[0] - Ph // 2, 0), min(center[0] + Ph // 2, H)
        min_w, max_w = max(center[1] - Pw // 2, 0), min(center[1] + Pw // 2, W)
        min_d, max_d = max(center[2] - Pd // 2, 0), min(center[2] + Pd // 2, D)

        # Check if coordinates are valid
        assert min_h < max_h, "Invalid coordinates: min_h >= max_h"
        assert min_w < max_w, "Invalid coordinates: min_w >= max_w"
        assert min_d < max_d, "Invalid coordinates: min_d >= max_d"

        # Crop the patch from the image
        patch = img[:, min_h:max_h, min_w:max_w, min_d:max_d]

        return patch

from monai.transforms import MapTransform, Transform
from monai import transforms as monai_transforms
import torchvision
from monai.config.type_definitions import NdarrayOrTensor
from typing import Any, Dict, Hashable, Mapping, Tuple
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import rotate


def random_crop2d(img, crop_size=50, jitter=0):
    """
    Randomly crop a 2D image to the specified size with random jitter if specified.
    Args:
        img (numpy.ndarray): The input image.
        crop_size (int): The size of the crop.
        jitter (int): The jitter value for random cropping.
    Returns:
        img_2d (numpy.ndarray): The cropped 2D image.
    """
    center_ind = np.array(img.shape) // 2
    crop_start = center_ind - crop_size // 2
    if jitter != 0:
        crop_start = crop_start + np.random.randint(-jitter, jitter, 2)
    crop_end = crop_start + crop_size
    img_2d = img[crop_start[0] : crop_end[0], crop_start[1] : crop_end[1]]
    assert img_2d.shape == (crop_size, crop_size)
    return img_2d


def slices2d_9(img, jitter=0):
    """
    Get 9 slices from the 3D image in 9 different directions.
    Args:
        img (numpy.ndarray): The input 3D image.
    Returns:
        nine_directions (list): A list of 9 cropped 2D images.
    """
    nine_directions = []
    center_ind = [img.shape[0] // 2] * 9
    # axial
    nine_directions.append(img[center_ind[0], :, :])
    nine_directions.append(img[:, center_ind[1], :])
    nine_directions.append(img[:, :, center_ind[2]])

    # rotate 45 degree
    rotated_img_02 = rotate(img, 45, axes=(0, 2), reshape=False)
    rotated_img_12 = rotate(img, 45, axes=(1, 2), reshape=False)
    rotated_img_01 = rotate(img, 45, axes=(0, 1), reshape=False)
    nine_directions.append(rotated_img_02[center_ind[3], :, :])
    nine_directions.append(rotated_img_02[:, :, center_ind[4]])
    nine_directions.append(rotated_img_01[center_ind[5], :, :])
    nine_directions.append(rotated_img_01[:, center_ind[6], :])
    nine_directions.append(rotated_img_12[:, center_ind[7], :])
    nine_directions.append(rotated_img_12[:, :, center_ind[8]])
    nine_directions = [
        convert_to_rgb(resize_img(random_crop2d(img_i, 50, jitter)))
        for img_i in nine_directions
    ]

    return nine_directions


def resize_img(img, target_shape=(224, 224)):
    """
    Resize the input image to the target shape using linear interpolation.
    Args:
        img (numpy.ndarray): The input image.
        target_shape (tuple): The target shape to resize the image to.
    Returns:
        numpy.ndarray: The resized image."""
    # Calculate the zoom factors for each dimension
    zoom_factors = [target_shape[i] / img.shape[i] for i in range(2)]
    return zoom(img, zoom_factors, order=1)


def convert_to_rgb(img):
    """
    Convert a single-channel image to a 3-channel RGB image by repeating the channel.
    Args:
        img (numpy.ndarray): The input single-channel image.
    Returns:
        numpy.ndarray: The converted 3-channel RGB image.
    """
    img = np.stack((img,) * 3, axis=0)
    return img


"""
The follwing function is adapted from foundation cancer image biomarker model code
https://github.com/AIM-Harvard/foundation-cancer-image-biomarker
"""


def get_transforms_raw(spatial_size=(50, 50, 50), jitter=None):
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
        start_h, end_h = center[0] - Ph // 2, center[0] + (Ph + 1) // 2
        start_w, end_w = center[1] - Pw // 2, center[1] + (Pw + 1) // 2
        start_d, end_d = center[2] - Pd // 2, center[2] + (Pd + 1) // 2

        min_h, max_h = max(start_h, 0), min(end_h, H)
        min_w, max_w = max(start_w, 0), min(end_w, W)
        min_d, max_d = max(start_d, 0), min(end_d, D)

        # min_h, max_h = max(center[0] - Ph // 2, 0), min(center[0] + Ph // 2, H)
        # min_w, max_w = max(center[1] - Pw // 2, 0), min(center[1] + Pw // 2, W)
        # min_d, max_d = max(center[2] - Pd // 2, 0), min(center[2] + Pd // 2, D)

        # Check if coordinates are valid
        assert min_h < max_h, "Invalid coordinates: min_h >= max_h"
        assert min_w < max_w, "Invalid coordinates: min_w >= max_w"
        assert min_d < max_d, "Invalid coordinates: min_d >= max_d"

        # Crop the patch from the image
        patch = img[:, min_h:max_h, min_w:max_w, min_d:max_d]

        # pad
        pad_h_before, pad_h_after = min_h - start_h, end_h - max_h
        pad_w_before, pad_w_after = min_w - start_w, end_w - max_w
        pad_d_before, pad_d_after = min_d - start_d, end_d - max_d

        pad = (
            (0, 0),
            (pad_h_before, pad_h_after),
            (pad_w_before, pad_w_after),
            (pad_d_before, pad_d_after),
        )
        patch_padded = np.pad(patch, pad, mode="constant", constant_values=0)

        return patch_padded

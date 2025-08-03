import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import SpatialResample
import random
import os
import torchio as tio

from collections import defaultdict
import clip
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from utils.preprocess import slices2d_9

import nlpaug.flow as naf
import nlpaug.augmenter.word as naw

# Change the following to your semantic features and label
SEMANTIC_FEATS = [
    "nodule_margin_conspicuity",
    "nodule_margins",
    "additional_nodule_margins",
    "nodule_shape",
    "nodule_consistency",
    "nodule_reticulation",
    "cyst-like_spaces",
    "intra-nodular_bronchiectasis",
    "necrosis",
    "cavitation",
    "eccentric_calcification",
    "airway_cut-off",
    "pleural_attachment",
    "pleural_retraction",
    "vascular_convergence",
    "septal_stretching",
    "paracicatricial_emphysema",
]

LABEL = ["malignant"]

# Semantic features augmentation for synonyms replacement
reserved_tokens = [
    ["mm", "millimeter"],
    ["attached", "adhered"],
    ["attachment", "adhesion"],
    ["presence", "existence", "appearance"],
    ["border", "margin", "boundary"],
    [
        "displays",
        "shows",
        "exhibits",
        "presents",
        "demonstrates",
        "reveals",
        "showcases",
    ],
    ["greatest", "longest", "maximum"],
    ["shortest", "smallest", "minimum"],
    ["observed", "seen", "evident", "noted"],
    ["presence", "evidence", "signs"],
    ["absence", "no evidence", "no signs"],
    ["not attached", "unattached"],
    ["retraction", "indentation"],
    ["around", "in the vicinity of", "surrounding", "adjacent to", "near"],
]


class CLIPDatasetText(Dataset):
    """
    Dataset class for loading vision and text data during training
    """

    def __init__(self, args, fold, mode="train", seed=0):
        super().__init__()
        self.resampler = SpatialResample(mode="bilinear")
        self.dataset_path = args.dataset_path
        self.img_dir = args.img_dir
        self.text_dir = args.text_dir
        self.split_dir = args.split_dir
        self.mode = mode
        self.clip_min = args.clip_min
        self.clip_max = args.clip_max
        self.crop_size = args.crop_size

        # augmentation
        self.augmentation = args.augmentation
        self.random_flip_prob = args.random_flip_prob
        self.random_affine_degree = args.random_affine_degree
        self.random_noise_std = args.random_noise_std
        self.random_noise_mean = args.random_noise_mean
        self.random_gamma = args.random_gamma

        self.reverse_max = args.reverse_max
        self.random_crop_prob = args.random_crop_prob

        if mode == "train":
            self.jitter = args.jitter
            if self.augmentation:
                print("Augmentation is on.")
            else:
                print("Augmentation is off.")
        else:
            self.jitter = 0

        # load data
        try:
            dataset_csv = pd.read_csv(args.dataset_path)
            dataset_csv["pid"] = dataset_csv["pid"].astype(str)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset CSV file not found at {args.dataset_path}"
            )

        # load the split
        data_subset = self.load_split(args.split_dir, fold, dataset_csv, mode)
        data_subset = data_subset.reset_index(drop=True)

        # get weights based on class and semantic features
        self.class_weights, self.sample_weights = self.get_weights(data_subset)
        self.semantic_weights = self.get_semantic_weights(data_subset)

        self.data_subset = data_subset
        print(f"dataset {mode} size: {data_subset.shape[0]}")

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def __getitem__(self, idx):
        pid = self.data_subset.pid[idx]
        nodule_id = self.data_subset.nodule_id[idx]
        label = self.data_subset[LABEL].values[idx]

        img_path = f"{self.img_dir}/{pid}_{nodule_id}.pt"
        img = torch.load(img_path).squeeze().numpy()

        # clip to -1000 to 500
        img = np.clip(img, self.clip_min, self.clip_max)

        # normalize
        img = (img - self.clip_min) / (self.clip_max - self.clip_min)

        # augmentation
        if self.mode == "train" and self.augmentation:
            img = self.augment(img)

        # crop and slice from 9 directions
        nine_slices = slices2d_9(img)
        img_2d = torch.stack(
            [self.normalize(torch.from_numpy(s)) for s in nine_slices], dim=0
        )

        with open(f"{self.text_dir}/{pid}_{nodule_id}.txt") as f:
            text = f.read()

        return img_2d, text, label, pid, nodule_id

    def load_split(self, split_dir, fold, dataset_csv, mode):
        """
        Load the split based on the fold and mode.
        Args:
            split_dir (str): Directory containing the split files.
            fold (int): Fold number.
            dataset_csv (pd.DataFrame): DataFrame containing the dataset.
            mode (str): Mode of the dataset ('train', 'val').]
        Returns:
            pd.DataFrame: Subset of the dataset based on the split.
        """
        try:
            pid = pd.read_csv(
                os.path.join(split_dir, f"fold_{fold}", f"{mode}_pid.csv")
            ).astype(str)
            dataset_csv_sub = dataset_csv[dataset_csv["pid"].isin(pid["pid"])]
        except:
            raise ValueError(f"split fold {fold} not found")
        return dataset_csv_sub

    def __len__(self):
        return len(self.data_subset)

    def augment(self, img):
        """
        Apply augmentation to the image
        Args:
            img (np.ndarray): Image to augment.
        Returns:
            np.ndarray: Augmented image.
        """
        image = tio.ScalarImage(tensor=img[None])  # Add channel dimension
        transform = tio.Compose(
            [
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=self.random_flip_prob),
                tio.RandomAffine(degrees=self.random_affine_degree),
                tio.RandomNoise(mean=self.random_noise_mean, std=self.random_noise_std),
                tio.RandomGamma((-self.random_gamma, self.random_gamma)),
            ]
        )
        augmented_image = transform(image)
        augmented_image_np = augmented_image.numpy()[0]
        return augmented_image_np

    @staticmethod
    def get_weights(semantic_features_subset):
        """
        Get class weights and sample weights based on the labels.
        Args:
            semantic_features_subset (pd.DataFrame): DataFrame containing the semantic features.
        Returns:
            tuple: Class weights and sample weights.
        """
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(semantic_features_subset[LABEL])
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts.astype(float)
        class_weights = class_weights / class_weights.sum()
        sample_weights = class_weights[labels]

        return class_weights, sample_weights

    @staticmethod
    def get_semantic_weights(semantic_features_subset, eps=1e-6):
        """
        Get semantic weights based on the semantic features.
        Args:
            semantic_features_subset (pd.DataFrame): DataFrame containing the semantic features.
        Returns:
            np.ndarray: Normalized semantic weights.
        """
        weights = []
        for feat in SEMANTIC_FEATS:
            label_encoder = LabelEncoder()
            most_occur = semantic_features_subset[feat].value_counts().index[0]
            labels = label_encoder.fit_transform(
                semantic_features_subset[feat].fillna(most_occur)
            )
            class_counts = np.bincount(labels)
            class_weights = 1.0 / class_counts.astype(float)
            sample_weights = np.log(class_weights[labels])
            weights.append(sample_weights)
        weights = np.sum(np.stack(weights), axis=0)
        normalized_weights = (weights - np.min(weights) + eps) / (
            np.max(weights) - np.min(weights)
        )
        return normalized_weights


class CLIPDatasetTextCollator:
    """
    Collator for CLIPDatasetText to prepare batches for training.
    It handles the augmentation of text data and prepares the inputs for the model.

    Args:
        args (argparse.Namespace): Arguments containing the configuration for the dataset.
        mode (str): Mode of the dataset ('train', 'val'). Defaults to 'train

    Returns:
        dict: A dictionary containing the inputs for the model, including pixel values, input IDs,
                labels, pids, and nodule IDs.
    """

    def __init__(self, args, mode="train"):
        self.args = args
        self.mode = mode
        if mode == "train":
            self.aug = self.aug_fun()

    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        for data in batch:

            # image
            inputs["pixel_values"].append(data[0])

            # text
            text = data[1]
            pos_i = text.lower().find("impression")
            texts = [text[:pos_i].replace("\n", "."), text[pos_i:].replace("\n", "")]

            if self.mode == "train":
                # randomly select one of the two texts
                text_selected = texts[-1]
                if random.random() < 0.5:
                    text_selected = self.aug.augment(text_selected)[0]
            else:
                text_selected = texts[-1]

            report_list.append(text_selected)

            inputs["labels"].append(data[2])
            inputs["pids"].append(data[3])
            inputs["nodule_ids"].append(data[4])

        text_inputs = clip.tokenize(report_list, truncate=True)
        inputs["input_ids"] = text_inputs
        inputs["pixel_values"] = torch.stack(inputs["pixel_values"])
        inputs["labels"] = torch.tensor(np.stack(inputs["labels"]).astype(float))
        inputs["pixel_values3d"] = torch.stack(inputs["pixel_values3d"])

        return inputs

    def aug_fun(self):
        """
        Define the augmentation for text data.
        Returns:
            naf.Sometimes: Augmentation flow for text data.
        """
        reserved_aug = naw.ReservedAug(
            reserved_tokens=reserved_tokens, aug_max=self.args.reverse_max
        )
        crop_aug = naw.RandomWordAug(action="crop", aug_p=self.args.random_crop_prob)

        aug = naf.Sometimes(
            [
                reserved_aug,
                crop_aug,
            ]
        )

        return aug

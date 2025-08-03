import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from scipy.ndimage import zoom
from scipy.ndimage import rotate
from torchvision import transforms
from utils.preprocess import slices2d_9


class VisionDatasetText(Dataset):
    '''
    Dataset class for loading only images during inference
    '''
    def __init__(self, args):
        super().__init__()
        self.dataset_path = args.dataset_path
        self.img_dir = args.img_dir
        self.split_dir = args.split_dir
        self.jitter = args.jitter

        try:
            dataset_csv = pd.read_csv(args.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset CSV file not found at {args.dataset_path}")
        
        self.data_subset = dataset_csv
        print(f"dataset size: {dataset_csv.shape[0]}")

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def __getitem__(self, idx):
        pid = self.data_subset.pid[idx]
        nodule_id = self.data_subset.nodule_id[idx]

        img_path = f"{self.img_dir}/{pid}_{nodule_id}.pt"
        img = torch.load(img_path).squeeze().numpy()

        # clip to -1000 to 500
        img = np.clip(img, -1000, 500)
        # normalize
        img = (img + 1000) / 1500
        #crop and slice from 9 directions
        nine_slices = slices2d_9(img)
        img_2d = torch.stack(
            [self.normalize(torch.from_numpy(s)) for s in nine_slices], dim=0
        )

        return img_2d, pid, nodule_id

    def __len__(self):
        return len(self.data_subset)


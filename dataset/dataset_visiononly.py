
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

from scipy.ndimage import zoom
from scipy.ndimage import rotate
from torchvision import transforms

class VisionDatasetText(Dataset):
    def __init__(self, args):
        super().__init__()
        self.dataset_path = args.dataset_path
        self.img_dir = args.img_dir
        self.split_dir = args.split_dir
        self.jitter = args.jitter

        dataset_csv = pd.read_csv(args.dataset_path)

        self.data_subset = dataset_csv
        print(f"dataset size: {dataset_csv.shape[0]}")

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                                std=[0.26862954, 0.26130258, 0.27577711])

    def __getitem__(self, idx):
        pid = self.data_subset.pid[idx]
        nodule_id = self.data_subset.nodule_id[idx]

        img_path = f'{self.img_dir}/{pid}_{nodule_id}.pt'
        img = torch.load(img_path).squeeze().numpy()

        #clip to -1000 to 500
        img = np.clip(img, -1000, 500)
        #normalize
        img = (img + 1000) / 1500

        nine_slices = self.slices2d_9(img)
        img_2d = torch.stack([self.normalize(torch.from_numpy(s)) for s in nine_slices], dim=0)

        return img_2d
    
    
    def __len__(self):
        return len(self.data_subset)
    
    @staticmethod
    def random_crop3d(img, crop_size = 50, jitter = 0):
        center_ind = np.array(img.shape) // 2
        crop_start = center_ind - crop_size // 2
        if jitter != 0:
            crop_start = crop_start + np.random.randint(-jitter, jitter, 3)
        crop_end = crop_start + crop_size
        img_3d = img[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1], crop_start[2]:crop_end[2]]
        assert img_3d.shape == (crop_size, crop_size, crop_size)
        return img_3d
    
    @staticmethod
    def random_crop2d(img, crop_size = 50, jitter = 0):
        center_ind = np.array(img.shape) // 2
        crop_start = center_ind - crop_size // 2
        if jitter!=0:
            crop_start = crop_start + np.random.randint(-jitter, jitter, 2)
        crop_end = crop_start + crop_size
        img_2d = img[crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]
        assert img_2d.shape == (crop_size, crop_size)
        return img_2d
    
    def slices2d_9(self, img):
        nine_directions = []
        center_ind = [img.shape[0]//2] *9
        #axial
        nine_directions.append(img[center_ind[0], :, :])
        nine_directions.append(img[:, center_ind[1], :])
        nine_directions.append(img[:, :, center_ind[2]])

        #rotate 45 degree
        rotated_img_02 =  rotate(img, 45, axes=(0, 2), reshape=False)
        rotated_img_12 =  rotate(img, 45, axes=(1, 2), reshape=False)
        rotated_img_01 =  rotate(img, 45, axes=(0, 1), reshape=False)
        nine_directions.append(rotated_img_02[center_ind[3], :, :])
        nine_directions.append(rotated_img_02[:, :, center_ind[4]])
        nine_directions.append(rotated_img_01[center_ind[5], :, :])
        nine_directions.append(rotated_img_01[:, center_ind[6], :])
        nine_directions.append(rotated_img_12[:, center_ind[7], :])
        nine_directions.append(rotated_img_12[:, :, center_ind[8]])
        nine_directions= [self.convert_to_rgb(self.resize_img(self.random_crop2d(img_i, 50, 0))) for img_i in nine_directions]

        return nine_directions

    @staticmethod
    def resize_img(img, target_shape  = (224, 224)):
        # Calculate the zoom factors for each dimension
        zoom_factors = [target_shape[i] / img.shape[i] for i in range(2)]
        return zoom(img, zoom_factors, order=1)
    
    @staticmethod
    def convert_to_rgb(img):
        img = np.stack((img,)*3, axis=0)
        return img

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
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

import nlpaug.flow as naf
import nlpaug.augmenter.word as naw

# Change the following to your semantic features and label
SEMANTIC_FEATS = ['nodule_margin_conspicuity',
       'nodule_margins', 'additional_nodule_margins', 'nodule_shape',
       'nodule_consistency', 'nodule_reticulation', 'cyst-like_spaces',
       'intra-nodular_bronchiectasis', 'necrosis', 'cavitation',
       'eccentric_calcification', 'airway_cut-off', 'pleural_attachment',
       'pleural_retraction', 'vascular_convergence', 'septal_stretching',
       'paracicatricial_emphysema']

LABEL = ['malignant']

#Semantic features augmentation for synonyms replacement
reserved_tokens = [
    ['mm', 'millimeter'],
    ['attached','adhered'],
    ['attachment','adhesion'],
    ['presence','existence','appearance'],
    ['border','margin','boundary'],
    ['displays','shows','exhibits','presents','demonstrates','reveals','showcases'],
    ['greatest','longest','maximum'],
    ['shortest','smallest','minimum'],
    ['observed','seen','evident','noted'],
    ['presence','evidence', 'signs'],
    ['absence','no evidence', 'no signs'],
    ['not attached','unattached'],
    ['retraction','indentation'],
    ['around', 'in the vicinity of', 'surrounding', 'adjacent to', 'near']

]

class CLIPDatasetText(Dataset):
    def __init__(self, args, fold,  mode = 'train',seed = 0):
        super().__init__()
        self.resampler = SpatialResample( mode='bilinear')
        self.dataset_path = args.dataset_path
        self.img_dir = args.img_dir
        self.text_dir = args.text_dir
        self.split_dir = args.split_dir
        self.mode = mode
        self.clip_min = args.clip_min
        self.clip_max = args.clip_max
        self.crop_size = args.crop_size

        #augmentation
        self.augmentation = args.augmentation
        self.random_flip_prob = args.random_flip_prob
        self.random_affine_degree = args.random_affine_degree
        self.random_noise_std = args.random_noise_std
        self.random_noise_mean = args.random_noise_mean
        self.random_gamma = args.random_gamma

        self.reverse_max = args.reverse_max
        self.random_crop_prob = args.random_crop_prob

        if mode == 'train':
            self.jitter = args.jitter
            if self.augmentation:
                print('Augmentation is on.')
            else:
                print('Augmentation is off.')
        else:
            self.jitter = 0

        #load data
        dataset_csv = pd.read_csv(args.dataset_path)

        # load the split
        data_subset = self.load_split(args.split_dir, fold, dataset_csv, mode)
        data_subset = data_subset.reset_index(drop = True)

        # get weights based on class and semantic features
        self.class_weights, self.sample_weights = self.get_weights(data_subset)
        self.semantic_weights = self.get_semantic_weights(data_subset)
        
        self.data_subset = data_subset
        print(f"dataset {mode} size: {data_subset.shape[0]}")

        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])


    def __getitem__(self, idx):
        pid = self.data_subset.pid[idx]
        nodule_id = self.data_subset.nodule_id[idx]
        label = self.data_subset.malignancy[idx]


        img_path = f'{self.img_dir}/{pid}_{nodule_id}.pt'
        img = torch.load(img_path).squeeze().numpy()

        #clip to -1000 to 500
        img = np.clip(img, self.clip_min, self.clip_max)

        #normalize
        img = (img - self.clip_min) / (self.clip_max - self.clip_min)

        #augmentation
        if self.mode == 'train' and self.augmentation:
            img = self.augment(img)

        #random crop 
        img_3d = self.random_crop3d(img, self.crop_size, jitter = self.jitter)

        #slice from 9 directions
        nine_slices = self.slices2d_9(img)
        img_2d = torch.stack([self.normalize(torch.from_numpy(s)) for s in nine_slices], dim=0)

        with open(f'{self.text_dir}/{pid}_{nodule_id}.txt') as f:
            text = f.read()

        return torch.from_numpy(img_3d).unsqueeze(0), img_2d, text, label, pid, nodule_id

    def load_split(self, split_dir, fold, dataset_csv, mode):
        try:
            pid =pd.read_csv(os.path.join(split_dir,f'fold_{fold}', f'{mode}_pid.csv')).astype(str)
            dataset_csv_sub = dataset_csv[dataset_csv['pid'].isin(pid['pid'])]
        except:
            raise ValueError(f"split fold {fold} not found")
        return dataset_csv_sub
    
    def __len__(self):
        return len(self.data_subset)
    

    def augment(self, img):
        image = tio.ScalarImage(tensor=img[None])  # Add channel dimension
        transform = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=self.random_flip_prob),
            tio.RandomAffine(degrees=self.random_affine_degree),
            tio.RandomNoise(mean=self.random_noise_mean, std=self.random_noise_std),
            tio.RandomGamma((-self.random_gamma, self.random_gamma)),
        ])
        augmented_image = transform(image)
        augmented_image_np = augmented_image.numpy()[0]
        return augmented_image_np
    
    @staticmethod
    def random_crop3d(img, crop_size = 50, jitter = 0):
        center_ind = np.array(img.shape) // 2
        crop_start = center_ind - crop_size // 2
        if jitter!=0:
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
        #axial
        if self.mode == 'train':
            center_ind = [img.shape[0]//2] *9 + np.random.randint(-3, 3, 9)
        else:
            center_ind = [img.shape[0]//2] *9
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

        nine_directions= [self.convert_to_rgb(self.resize_img(self.random_crop2d(img_i, 50, self.jitter))) for img_i in nine_directions]

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
    
    @staticmethod
    def get_weights( semantic_features_subset):
        label_encoder = LabelEncoder()
        import pdb; pdb.set_trace()
        most_occur = semantic_features_subset[LABEL].value_counts().index[0]
        labels = label_encoder.fit_transform(semantic_features_subset[LABEL].fillna(most_occur))
        class_counts = np.bincount(labels)
        class_weights = 1.0/class_counts.astype(float)
        class_weights = class_weights/class_weights.sum()
        sample_weights = class_weights[labels]

        return class_weights, sample_weights
    
    @staticmethod
    def get_semantic_weights(semantic_features_subset):
        eps = 1e-6
        weights = []
        for feat in SEMANTIC_FEATS:
            label_encoder = LabelEncoder()
            most_occur = semantic_features_subset[feat].value_counts().index[0]
            labels = label_encoder.fit_transform(semantic_features_subset[feat].fillna(most_occur))
            class_counts = np.bincount(labels)
            class_weights = 1.0/class_counts.astype(float)
            sample_weights = np.log(class_weights[labels])
            weights.append(sample_weights)
        weights = np.sum(np.stack(weights),axis =0)
        normalized_weights = (weights - np.min(weights) + eps) / (np.max(weights) - np.min(weights))
        return normalized_weights


class CLIPDatasetTextCollator:
    def __init__(self, args, mode = 'train'):
        self.args = args
        self.mode = mode
        if mode == 'train':
            self.aug = self.aug_fun()

    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        for data in batch:

            #image
            inputs['pixel_values'].append(data[1])
            inputs['pixel_values3d'].append(data[0])

            #text
            text = data[2]
            pos_i = text.lower().find("impression")
            texts = [text[:pos_i].replace('\n','.'), text[pos_i:].replace('\n','')]
            
            if self.mode == 'train':
                #randomly select one of the two texts
                text_selected = texts[-1]
                if random.random() < 0.5:
                    text_selected = self.aug.augment(text_selected)[0]
            else:
                text_selected = texts[-1]

            report_list.append(text_selected)

            inputs['labels'].append(data[3])
            inputs['pids'].append(data[4])
            inputs['nodule_ids'].append(data[5])


        text_inputs = clip.tokenize(report_list, truncate=True)
        inputs['input_ids'] = text_inputs
        inputs['attention_mask'] = torch.ones_like(text_inputs)


        inputs['pixel_values'] = torch.stack(inputs['pixel_values'])
        inputs['labels'] = torch.tensor(np.stack(inputs['labels']).astype(float))
        inputs['pixel_values3d'] =torch.stack(inputs['pixel_values3d'])


        return inputs
    
    def aug_fun(self):


        reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens,  aug_max = self.args.reverse_max)
        crop_aug = naw.RandomWordAug(action='crop', aug_p = self.args.random_crop_prob)


        aug = naf.Sometimes([
            reserved_aug,
            crop_aug,
        ])

        return aug
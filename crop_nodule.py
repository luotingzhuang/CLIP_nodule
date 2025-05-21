import numpy as np
import torch
import pandas as pd
import os
import monai
from tqdm import tqdm
from monai.data import CSVDataset
from utils.utils import get_transforms_raw



class NoduleDataset(CSVDataset):
    def __init__(self, dataset_path, subset = None):

        self.dataset_path = dataset_path
        self.transform = get_transforms_raw(spatial_size=(100, 100, 100))
        dataset_csv = pd.read_csv(dataset_path)
        self.dataset_csv = dataset_csv
        super().__init__(dataset_csv, transform=self.transform)

        
    def __getitem__(self, idx):
        pid = self.dataset_csv.pid[idx]
        nodule_id = self.dataset_csv.nodule_id[idx]
        img_processed = super().__getitem__(idx)

        return pid, nodule_id, img_processed


    def __len__(self):
        return len(self.dataset_csv)

def loadargs():
    '''
    Load the arguments for evaluation.
    '''
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_path', type=str, default='../cropped_img', help='Path to save the cropped nodules')
    return parser.parse_args()
    

if __name__ == "__main__":
    args = loadargs()
    dataset = NoduleDataset(args.dataset_path, subset = 'DLCS')
    dataloader = monai.data.DataLoader(dataset, 
                        batch_size=1, 
                        num_workers=args.num_workers, 
                        pin_memory=False,
                        shuffle=False)
        
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader)):
            pid = batch[0][0]
            nodule_id = batch[1].item()
            img = batch[2]

            save_pt = f'{args.save_path}/{pid}_{nodule_id}.pt'
            if os.path.exists(save_pt):
                continue
            torch.save(img, save_pt)

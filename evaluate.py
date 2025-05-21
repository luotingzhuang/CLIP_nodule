import torch
import pandas as pd
import os
import monai
import argparse

from utils.utils import init_model
from dataset.dataset_visiononly import VisionDatasetText
from utils.inference import load_args, eval_epoch


def load_eval_args():
    '''
    Load the arguments for evaluation.
    '''
    parser = argparse.ArgumentParser(description='Evaluate the model')
    parser.add_argument('--model_path', type=str, default='./ckpt', help='Path to the model')
    parser.add_argument('--dataset_path', type=str, default='./dataset_csv/sample_csv.csv', help='Path to the dataset')
    parser.add_argument('--img_dir', type=str, default='./cropped_img', help='Path to the cropped images')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_path', type=str, default='./results_csv', help='Path to save the results')
    return parser.parse_args()


if __name__ == '__main__':
    eval_args = load_eval_args()
    args = load_args(eval_args.model_path)
    args.dataset_path =  eval_args.dataset_path
    args.img_dir = eval_args.img_dir

    # DataLoaders
    test_dataset = VisionDatasetText(args)
    test_loader = monai.data.DataLoader(test_dataset, 
                                        batch_size=1, 
                                        num_workers=eval_args.num_workers, 
                                        shuffle =False,
                                        pin_memory=False)

    model = init_model(args)
    all_test_result = []

    #Loop through folds
    for fold in range(5):
        print(f'Loading fold {fold}...')
        try:
            weight_path = os.path.join(eval_args.model_path,f'fold_{fold}/best_both.pt')
            print('Loading model from', weight_path)
            pretrained_dict = torch.load(weight_path, weight_only=False)['model']
        except:
            print('Model path does not exist. Exiting...')
            exit()
        
        model.load_state_dict(pretrained_dict, strict=False)
        model.eval()
        with torch.no_grad():
            probs = eval_epoch(model, test_loader)
            result = test_dataset.data_subset[['pid']].copy()
            result.loc[:,'probs'] = probs[:,1]
            result.loc[:,'fold'] = fold
            all_test_result.append(result)

    # Organize results from all folds
    outputdf = {}
    for i in range(5):
        all_test_result_i = all_test_result[i].groupby('pid').max().reset_index()
        outputdf[f'raw_{i}'] =  all_test_result_i.probs.values
            
    outputdf = pd.DataFrame(outputdf)
    outputdf.loc[:,'ensemble'] = outputdf.mean(1).values
    outputdf.loc[:,'pid'] = all_test_result_i.pid.values

    os.makedirs(eval_args.save_path, exist_ok=True)
    outputdf.to_csv(os.path.join(eval_args.save_path, 'result.csv'), index=False)
    print('Results saved to', os.path.join(eval_args.save_path, 'result.csv'))
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
    #'./results_newfreeze/modelopenai_ViT-B_32_tuninglora_bz16_j5_lr0.0001_wd0.1_epochs100_ga1_patience20_weightedsemantic_tau0.03_clipweight1.0_imgweight1.0_textweight1.0_lora_all_both_r2_alpha1_dropout0.25_paramsqkv'
    parser.add_argument('--dataset_path', type=str, default='./dataset_csv/datasets_09262024.csv', help='Path to the dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_path', type=str, default='./results_csv', help='Path to save the results')
    return parser.parse_args()


if __name__ == '__main__':
    eval_args = load_eval_args()
    args = load_args(eval_args.model_path)
    args.dataset_path =  eval_args.dataset_path

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
        try:
            weight_path = os.path.join(eval_args.model_path,f'fold_{fold}/best_both.pt')
            pretrained_dict = torch.load(weight_path)['model']
        except:
            print('Model path does not exist. Exiting...')
            exit()
        
        model.load_state_dict(pretrained_dict)
        model.eval()
        with torch.no_grad():
            probs = eval_epoch(model, test_loader)
            result = test_dataset.data_subset[['pid']]
            result['probs'] = probs[:,1]
            result['fold'] = fold
            all_test_result.append(result)


    # Organize results from all folds
    outputdf = {}
    for i in range(5):
        all_test_result_i = all_test_result[i].groupby('pid').max().reset_index()
        outputdf[f'raw_{i}'] =  all_test_result_i.probs
            
    outputdf = pd.DataFrame(outputdf)
    outputdf['ensemble'] = outputdf.mean(1)
    outputdf['pid'] = all_test_result_i['pid']
    outputdf.to_csv(os.path.join(eval_args.save_path, 'ensemble.csv'), index=False)
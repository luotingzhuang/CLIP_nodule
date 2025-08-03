import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler

from utils.utils import init_model, build_loaders, EarlyStopping
from utils.train import train_epoch, valid_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_args():
    #load arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./dataset_csv/datasets.csv", help="Path to the dataset csv file")
    parser.add_argument("--result_dir", type=str, default="./results", help="Path to the result directory")
    parser.add_argument("--text_dir", type=str, default="../report_generation", help="Path to the report directory")
    parser.add_argument("--img_dir", type=str, default="../cropped_img", help="Path to the image directory")
    parser.add_argument("--split_dir", type=str, default="./splits/split_fold5_seed0", help="Path to the split directory")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits")
    parser.add_argument("--k_start", type=int, default=0, help="Start fold for cross-validation")
    parser.add_argument("--k_end", type=int, default=5, help="End fold for cross-validation")
    parser.add_argument("--debug", action='store_true',default=False, help='debug mode')
    parser.add_argument("--model", type=str, default="openai_ViT-B/32", choices=["openai_ViT-B/32"], help="CLIP Model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-1, help="Weight decay for the optimizer")
    parser.add_argument("--patience", type=int, default=5, help="Patience for the learning rate scheduler")
    parser.add_argument("--factor", type=float, default=0.5, help="Factor for the learning rate scheduler")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--es_warmup", type=int, default=0, help="Warmup epochs for early stopping")
    parser.add_argument("--es_patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--jitter", type=int, default=5, help="Jitter for data augmentation")
    parser.add_argument("--augmentation", action='store_true',default=False, help='Use data augmentation')
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--ga", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--tuning", type=str, default = 'ft', choices = ['ft', 'pt','lora'], help="Tuning method")
    parser.add_argument('--clip_loss_weight', type=float, default=1.0, help="Weight for the CLIP loss")
    parser.add_argument('--img_loss_weight', type=float, default=1.0, help="Weight for the image loss")
    parser.add_argument('--text_loss_weight', type=float, default=1.0, help="Weight for the text loss")
    parser.add_argument('--weighted', type=str, default='diagnosis', choices = ['diagnosis', 'semantic'], help="Weighted sampling method")
    parser.add_argument('--tau', type=float, default=0.07, help="Temperature for the CLIP loss")

    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+',default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') #['qkv','query','key','value']
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')

    
    args = parser.parse_args()
    return args

def main():
    args = load_args()
    result_dir = args.result_dir
    exp_name = f"model{args.model.replace('/','_')}_tuning{args.tuning}_bz{args.batch_size}_j{args.jitter}_lr{args.lr}_wd{args.weight_decay}_epochs{args.epochs}_ga{args.ga}_patience{args.es_patience}_weighted{args.weighted}_tau{args.tau}"
    exp_name += f"_clipweight{args.clip_loss_weight}_imgweight{args.img_loss_weight}_textweight{args.text_loss_weight}"

    if 'lora' in args.tuning:
        exp_name += f"_lora_{args.position}_{args.encoder}_r{args.r}_alpha{args.alpha}_dropout{args.dropout_rate}"
        exp_name += f"_params{''.join(args.params)}"


    result_dir = os.path.join(result_dir, exp_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        print("Folder already exists, do you want to overwrite?")
        overwrite = input("Overwrite? (y/n)")
        if overwrite.lower() != 'y':
            return

    #save args
    with open(os.path.join(result_dir, "args.txt"), "w") as f:
        f.write(str(args))
    

    for fold in range(args.k_start, args.k_end):
        print(f"---------------Fold: {fold}---------------")
        result_dir_fold = os.path.join(result_dir, f"fold_{fold}")
            
        writer = SummaryWriter(log_dir= result_dir_fold)
        early_stopping_clip = EarlyStopping(
                    warmup=args.es_warmup, patience=args.es_patience, verbose=True
                )
        early_stopping_pred = EarlyStopping(
                    warmup=args.es_warmup, patience=args.es_patience, verbose=True
                )
        early_stopping_both = EarlyStopping(
                    warmup=args.es_warmup, patience=args.es_patience, verbose=True
                )
        
        train_loader = build_loaders(args, fold = fold, mode="train")
        valid_loader = build_loaders(args, fold = fold, mode="val")
        test_loader = build_loaders(args, fold = fold, mode="test")

        # init model and computes total trainable weights
        model = init_model(args)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Trainable parameters: {total_trainable_params}/{total_params}.')

        # optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        for epoch in range(args.epochs):
            print(f"Epoch: {epoch + 1}")
            model.train() 
            train_loss, train_clip_loss, train_pred_loss, train_acc_img, train_acc_sem, train_auc_img, train_auc_text, train_aupuc_img, train_aupuc_text = train_epoch(args, model, train_loader, optimizer)
            writer.add_scalar('Loss/train', train_loss.avg, epoch)
            writer.add_scalar('Loss/train_clip', train_clip_loss.avg, epoch)
            writer.add_scalar('Loss/train_pred', train_pred_loss.avg, epoch)
            writer.add_scalar('Acc/train_img', train_acc_img.avg, epoch)
            writer.add_scalar('Acc/train_semantic', train_acc_sem.avg, epoch)
            writer.add_scalar('AUC/train_img', train_auc_img, epoch)
            writer.add_scalar('AUC/train_semantic', train_auc_text, epoch)
            writer.add_scalar('AUPUC/train_img', train_aupuc_img, epoch)
            writer.add_scalar('AUPUC/train_semantic', train_aupuc_text, epoch)
            writer.add_scalar('logit_scale',model.logit_scale.exp().detach().cpu().item(), epoch)

            model.eval()
            with torch.no_grad():
                valid_loss, val_clip_loss, val_pred_loss, val_acc_img, val_acc_sem, val_auc_img, val_auc_text, val_aupuc_img, val_aupuc_text = valid_epoch(args, model, valid_loader, mode = 'val')
                writer.add_scalar('Loss/valid', valid_loss.avg, epoch)
                writer.add_scalar('Loss/valid_clip', val_clip_loss.avg, epoch)
                writer.add_scalar('Loss/valid_pred', val_pred_loss.avg, epoch)
                writer.add_scalar('Acc/valid_img', val_acc_img.avg, epoch)
                writer.add_scalar('Acc/valid_semantic', val_acc_sem.avg, epoch)
                writer.add_scalar('AUC/valid_img', val_auc_img, epoch)
                writer.add_scalar('AUC/valid_semantic', val_auc_text, epoch)
                writer.add_scalar('AUPUC/valid_img', val_aupuc_img, epoch)
                writer.add_scalar('AUPUC/valid_semantic', val_aupuc_text, epoch)


            # save best model based on validation loss
            if args.img_loss_weight > 0 or args.text_loss_weight > 0:
                early_stop_pred = early_stopping_pred(epoch=epoch,
                                        val_loss=val_pred_loss.avg,
                                        model=model,
                                        ckpt_path=os.path.join(result_dir_fold,"best_pred.pt"))
            else:
                early_stop_pred = True

            if args.clip_loss_weight > 0:
                early_stop_clip = early_stopping_clip(epoch=epoch,
                                        val_loss=val_clip_loss.avg,
                                        model=model,
                                        ckpt_path=os.path.join(result_dir_fold,"best_clip.pt"))
                early_stop_both = early_stopping_both(epoch=epoch,
                                            val_loss=valid_loss.avg, 
                                            model=model,
                                            ckpt_path=os.path.join(result_dir_fold,"best_both.pt"))

            else:
                early_stop_clip = True
                early_stop_both = True

            #save the latest model
            state = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "early_stopping_pred": early_stopping_pred,
                "early_stopping_clip": early_stopping_clip,
                "early_stopping_both": early_stopping_both,

            }
            torch.save(state, os.path.join(result_dir_fold,f"ckpt.pt"))

            # early stop if all early stopping conditions are met
            if early_stop_clip & early_stop_pred & early_stop_both:
                print("Early Stopping")
                break

if __name__ == "__main__":

    main()

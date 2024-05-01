import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import monai
import config as CFG
from dataset import CLIPDataset
from CLIP import CLIPModel
from utils import AvgMeter, get_lr, EarlyStopping
from tensorboardX import SummaryWriter


def make_train_valid_dfs():
    dataframe = pd.read_csv(f"{CFG.captions_path}/captions.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(CFG, mode):
    dataset = CLIPDataset(
        CFG.coord_path, CFG.semantic_path, mode = mode
    )
    #if mode == "train":
    #    shuffle = True
    #else:
    #    shuffle = False
    dataloader = monai.data.DataLoader(dataset, 
                          batch_size=CFG.batch_size, 
                          num_workers=CFG.num_workers, 
                          shuffle=True)
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    acc_img_meter = AvgMeter()
    acc_semantic_meter = AvgMeter()

    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for ind, batch in enumerate(tqdm_object):
        loss, acc_img, acc_sem = model(batch)
        loss = loss / CFG.ga
        loss.backward()
        if (ind + 1) % CFG.ga == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if step == "batch":
            lr_scheduler.step()

        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        count = batch[0].size(0)
        loss_meter.update(loss.item(), count)
        acc_img_meter.update(acc_img, count)
        acc_semantic_meter.update(acc_sem, count)

        if (ind + 1) % CFG.ga == 0:
            print(f"Train Loss: {loss.item()}, Acc Img: {acc_img}, Acc Text: {acc_sem}, logit_scale: { model.logit_scale.exp()}")

        tqdm_object.set_postfix(train_loss=loss_meter.avg, train_acc_img=acc_img_meter.avg, 
                                train_acc_semantic=acc_semantic_meter.avg,lr=get_lr(optimizer))
    return loss_meter, acc_img_meter, acc_semantic_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    acc_img_meter = AvgMeter()
    acc_semantic_meter = AvgMeter()
    
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        loss, acc_img, acc_sem = model(batch)
        count = batch[0].size(0)
        loss_meter.update(loss.item(), count)
        acc_img_meter.update(acc_img, count)
        acc_semantic_meter.update(acc_sem, count)
        print(f"Val Loss: {loss.item()}, Acc Img: {acc_img}, Acc Text: {acc_sem}")
        tqdm_object.set_postfix(valid_loss=loss_meter.avg, val_acc_img=acc_img_meter.avg, val_acc_semantic=acc_semantic_meter.avg)
    return loss_meter, acc_img_meter, acc_semantic_meter



def main():

    result_path = CFG.result_path
    exp_name = f'bz{CFG.batch_size}_j{CFG.jitter}_lr{CFG.lr}_wd{CFG.weight_decay}_pd{CFG.projection_dim}_dropout{CFG.dropout}_epochs{CFG.epochs}_ga{CFG.ga}'
    if CFG.general:
        exp_name = f'{exp_name}_general'
    if CFG.internal:
        exp_name = f'{exp_name}_internal'
    if CFG.external:
        exp_name = f'{exp_name}_external'
    if CFG.freeze:
        exp_name = f'{exp_name}_freeze'
    result_path = os.path.join(result_path, exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        print("Folder already exists")
        return

    #save config
    #save into python file
    with open(os.path.join(result_path, 'config.py'), 'w') as f:
        for key, value in CFG.__dict__.items():
            if not key.startswith("__") and key != 'torch':
                f.write(f"{key} = {value}\n")

    writer = SummaryWriter(log_dir= result_path)
    early_stopping = EarlyStopping(
                warmup=CFG.es_warmup, patience=CFG.es_patience, verbose=True
            )

    train_loader = build_loaders(CFG, mode="train")
    valid_loader = build_loaders(CFG, mode="val")

    print('initialize model ...')
    semantic_embedding = train_loader.dataset.semantic_features.shape[1]
    print('image embedding:', CFG.image_embedding)
    print('semantic embedding: ', semantic_embedding)
    model = CLIPModel(semantic_embedding = semantic_embedding).to(CFG.device)
    print('model initialized')
    #computes total trainable weights

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameters: {total_trainable_params}/{total_params}.')


    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss, train_acc_img, train_acc_sem = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        writer.add_scalar('Loss/train', train_loss.avg, epoch)
        writer.add_scalar('Acc/train_img', train_acc_img.avg, epoch)
        writer.add_scalar('Acc/train_semantic', train_acc_sem.avg, epoch)

        model.eval()
        with torch.no_grad():
            valid_loss, val_acc_img, val_acc_sem = valid_epoch(model, valid_loader)
            writer.add_scalar('Loss/valid', valid_loss.avg, epoch)
            writer.add_scalar('Acc/valid_img', val_acc_img.avg, epoch)
            writer.add_scalar('Acc/valid_semantic', val_acc_sem.avg, epoch)
        

        early_stop = early_stopping(
                epoch=epoch,
                val_loss=valid_loss.avg,
                model=model,
                ckpt_path=os.path.join(result_path,"best.pt"),
            )
        
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "early_stopping": early_stopping,
        }

        torch.save(state, os.path.join(result_path,f"ckpt.pt"))

        if early_stop:
            print("Early Stopping")
            break

if __name__ == "__main__":

    main()

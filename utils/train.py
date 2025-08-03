from utils.utils import AvgMeter
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.utils import get_lr

def train_epoch(args, model, train_loader, optimizer ):
    loss_meter = AvgMeter()
    loss_clip_meter = AvgMeter()
    loss_pred_meter = AvgMeter()
    acc_img_meter = AvgMeter()
    acc_semantic_meter = AvgMeter()

    labels = []
    logits_img_all = []
    logits_text_all = []

    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for ind, batch in enumerate(tqdm_object):
        batch['mode'] = 'train'
        loss_clip, loss_pred, clip_acc_text, clip_acc_images, logits_img, logits_text = model(**batch)
        logits_img_all.append(logits_img.cpu().detach())
        logits_text_all.append(logits_text.cpu().detach())

        label = batch['labels'].cpu().detach()
        label = label[label >= 0]
        labels.append(label)

        loss = loss_clip + loss_pred
        loss = loss / args.ga
        loss.backward()
        if (ind + 1) % args.ga == 0:
            optimizer.step()
            optimizer.zero_grad()

        count = args.batch_size
        loss_meter.update(loss.item(), count)
        loss_clip_meter.update(loss_clip.item(), count)
        loss_pred_meter.update(loss_pred.item(), count)
        acc_img_meter.update(clip_acc_images, count)
        acc_semantic_meter.update(clip_acc_text, count)


        tqdm_object.set_postfix(train_loss=loss_meter.avg, 
                                train_loss_clip=loss_clip_meter.avg,
                                train_loss_pred=loss_pred_meter.avg,
                                lr=get_lr(optimizer),
                                train_acc_img=acc_img_meter.avg, 
                                train_acc_semantic=acc_semantic_meter.avg)
        
    labels = torch.cat(labels, dim = 0)
    logits_img_all = torch.cat(logits_img_all, dim = 0)
    logits_text_all = torch.cat(logits_text_all, dim = 0)

    prob_img = torch.softmax(logits_img_all, dim = -1)
    prob_text = torch.softmax(logits_text_all, dim = -1)

    auc_img = roc_auc_score(labels.cpu().numpy(), prob_img.cpu().detach().numpy()[:,1])
    auc_text = roc_auc_score(labels.cpu().numpy(), prob_text.cpu().detach().numpy()[:,1])
    aupuc_img = average_precision_score(labels.cpu().numpy(), prob_img.cpu().detach().numpy()[:,1])
    aupuc_text = average_precision_score(labels.cpu().numpy(), prob_text.cpu().detach().numpy()[:,1])
    print(f"Loss: {loss_meter.avg}, Acc Img: {acc_img_meter.avg}, Acc Text: {acc_semantic_meter.avg}")
    print(f"AUC Img: {auc_img}, AUC Text: {auc_text}, AUPUC Img: {aupuc_img}, AUPUC Text: {aupuc_text}")

    return loss_meter, loss_clip_meter, loss_pred_meter, acc_img_meter, acc_semantic_meter, auc_img, auc_text, aupuc_img, aupuc_text


def valid_epoch(args, model, valid_loader, mode = 'val'):
    loss_meter = AvgMeter()
    loss_clip_meter = AvgMeter()
    loss_pred_meter = AvgMeter()

    acc_img_meter = AvgMeter()
    acc_semantic_meter = AvgMeter()

    labels = []
    logits_img_all = []
    logits_text_all = []


    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch['mode'] = 'val'
        loss_clip, loss_pred, clip_acc_text, clip_acc_images, logits_img, logits_text = model(**batch)
        logits_img_all.append(logits_img.cpu().detach())
        logits_text_all.append(logits_text.cpu().detach())

        label = batch['labels'].cpu().detach()
        label = label[label >= 0]
        labels.append(label)


        loss = loss_clip + loss_pred        
        count = args.batch_size
        loss_meter.update(loss.item(), count)
        loss_clip_meter.update(loss_clip.item(), count)
        loss_pred_meter.update(loss_pred.item(), count)
        acc_img_meter.update(clip_acc_images, count)
        acc_semantic_meter.update(clip_acc_text, count)


        tqdm_object.set_postfix(valid_loss=loss_meter.avg, 
                                valid_loss_clip=loss_clip_meter.avg,
                                valid_loss_pred=loss_pred_meter.avg,
                                val_acc_img=acc_img_meter.avg, 
                                val_acc_semantic=acc_semantic_meter.avg,
                                logit_scale= model.logit_scale.exp().detach().cpu().item())
    
    labels = torch.cat(labels, dim = 0)
    logits_img_all = torch.cat(logits_img_all, dim = 0)
    logits_text_all = torch.cat(logits_text_all, dim = 0)

    prob_img = torch.softmax(logits_img_all, dim = -1)
    prob_text = torch.softmax(logits_text_all, dim = -1)

    auc_img = roc_auc_score(labels.cpu().numpy(), prob_img.cpu().detach().numpy()[:,1])
    auc_text = roc_auc_score(labels.cpu().numpy(), prob_text.cpu().detach().numpy()[:,1])
    aupuc_img = average_precision_score(labels.cpu().numpy(), prob_img.cpu().detach().numpy()[:,1])
    aupuc_text = average_precision_score(labels.cpu().numpy(), prob_text.cpu().detach().numpy()[:,1])
    print(f"Mode {mode} - Loss: {loss_meter.avg}, Acc Img: {acc_img_meter.avg}, Acc Text: {acc_semantic_meter.avg}")
    print(f"AUC Img: {auc_img}, AUC Text: {auc_text}, AUPUC Img: {aupuc_img}, AUPUC Text: {aupuc_text}")
    
    return loss_meter, loss_clip_meter, loss_pred_meter, acc_img_meter, acc_semantic_meter, auc_img, auc_text, aupuc_img, aupuc_text

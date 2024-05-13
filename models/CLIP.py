import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import config as CFG
from models.modules import ImageEncoder, TextEncoder, ProjectionHead


class CLIPModel(nn.Module):
    def __init__(
        self,
        #temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        semantic_embedding= None#CFG.semantic_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        if CFG.text:
            self.semantic_encoder = TextEncoder()

        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.semantic_projection = ProjectionHead(embedding_dim=semantic_embedding)
        #self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.7)))
        #self.loss_img = nn.CrossEntropyLoss()
        #self.loss_sem = nn.CrossEntropyLoss()

    def forward(self, batch):
        # Getting Image and Text Features
        img = batch[0].to(CFG.device)
        #semantic_features = batch[1].to(CFG.device)

        image_features = self.image_encoder(img)
        if CFG.text:
            semantic_features = self.semantic_encoder(batch[2].to(CFG.device), batch[3].to(CFG.device)) #input_ids, attention_mask
        else:
            semantic_features = batch[1].to(CFG.device)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        semantic_embeddings = self.semantic_projection(semantic_features)

        #normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        semantic_embeddings = semantic_embeddings / semantic_embeddings.norm(dim=-1, keepdim=True)

        # Calculating the Loss
        logit_scale = self.logit_scale.exp()
        logits = (semantic_embeddings @ image_embeddings.T) * logit_scale#/ self.temperature
        gt = torch.arange(logits.size(0)).to(CFG.device)
        #semantic_loss = self.loss_sem(logits, gt)
        #image_loss = self.loss_img(logits.T, gt)
        #loss = (semantic_loss + image_loss) / 2.0

        images_logits = image_embeddings @ image_embeddings.T
        texts_logits = semantic_embeddings @ semantic_embeddings.T

        targets = F.softmax(
            (images_logits + texts_logits) / 2 * logit_scale, dim=-1
        )
        print(targets)

        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)

        acc_sems = compute_accuracy(logits,gt)
        acc_images = compute_accuracy(logits.T,gt)

        return loss.mean(), acc_images, acc_sems


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def compute_accuracy(logits,gt):
    logits = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == gt).sum().item()
    total = gt.size(0)
    accuracy = correct / total
    return accuracy

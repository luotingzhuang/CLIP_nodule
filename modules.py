import torch
from torch import nn
import config as CFG

from fmcib.models import LoadModel, fmcib_model

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = fmcib_model()

    def forward(self, x):
        return self.model(x)



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

        #kaiming initialization
        nn.init.kaiming_normal_(self.projection.weight)
        nn.init.kaiming_normal_(self.fc.weight)

    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
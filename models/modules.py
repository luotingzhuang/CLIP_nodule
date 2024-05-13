import torch
from torch import nn
import config as CFG

from fmcib.models import LoadModel, fmcib_model

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self
    ):
        super().__init__()
        model = fmcib_model()
        
        if CFG.freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True
                
        self.model = model
    def forward(self, x):
        return self.model(x)


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.U = nn.Parameter(torch.randn(in_features, rank))
        self.V = nn.Parameter(torch.randn(rank, out_features))

    def forward(self, x):
        return torch.mm(torch.mm(x, self.U), self.V)


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        #if embedding_dim > 1000:
        #    self.projection = LowRankLinear(embedding_dim, projection_dim, projection_dim)
        #else:
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
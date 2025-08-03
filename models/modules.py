import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):

    """
    Projection Head for embedding projection in CLIP model.
    Args:
        embedding_dim (int): Dimension of the input embeddings.
        projection_dim (int): Dimension of the output projections.
        dropout (float): Dropout rate for the projection head.
    Returns:
        x (torch.Tensor): Projected embeddings after applying linear transformation, GELU activation, dropout"""
    def __init__(self, embedding_dim=512, projection_dim=256, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)

        # kaiming initialization
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


class Attention(nn.Module):
    """
    Attention-based Deep Multiple Instance Learning
    Link: https://arxiv.org/abs/1802.04712
    Implementation: https://github.com/AMLab-Amsterdam/AttentionDeepMIL

    Args:
        feature_size (int): Size of the input features.
        M (int): Dimension of the projected features.
        L (int): Dimension of the attention features.
        ATTENTION_BRANCHES (int): Number of attention branches.
    Returns:
        embed (torch.Tensor): Output tensor after applying attention mechanism.
    """

    def __init__(self, feature_size, M=256, L=128, ATTENTION_BRANCHES=1):
        super().__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = ATTENTION_BRANCHES
        self.feature_size = feature_size

        self.projector = nn.Sequential(
            nn.Linear(feature_size, self.M),
            nn.ReLU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES),  # matrix W
        )
        
        # kaiming initialization
        nn.init.kaiming_normal_(self.projector[0].weight)
        nn.init.kaiming_normal_(self.attention[0].weight)
        nn.init.kaiming_normal_(self.attention[2].weight)


    def forward(self, embeds):
        embeds = embeds.view(-1, self.feature_size)
        embeds = self.projector(embeds)  # [b x M]

        attentions = self.attention(embeds)  # [b x ATTENTION_BRANCHES]
        attentions = torch.transpose(attentions, 1, 0)  # [ATTENTION_BRANCHES x b]
        attentions = F.softmax(attentions, dim=1)  # softmax over b

        embed = torch.mm(attentions, embeds)  # [ATTENTION_BRANCHES x M]
        return embed

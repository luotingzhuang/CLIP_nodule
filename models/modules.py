import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
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
    """

    def __init__(self, feature_size, M=256, L=128, ATTENTION_BRANCHES=1):
        super().__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = 1
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

    def forward(self, h):
        h = h.view(-1, self.feature_size)
        h = self.projector(h)  # [b x M]

        a = self.attention(h)  # [b x ATTENTION_BRANCHES]
        a = torch.transpose(a, 1, 0)  # [ATTENTION_BRANCHES x b]
        a = F.softmax(a, dim=1)  # softmax over b

        z = torch.mm(a, h)  # [ATTENTION_BRANCHES x M]
        return z


class Attention_a(nn.Module):
    """
    Attention-based Deep Multiple Instance Learning
    Link: https://arxiv.org/abs/1802.04712
    Implementation: https://github.com/AMLab-Amsterdam/AttentionDeepMIL
    """

    def __init__(self, feature_size, M=256, L=128, ATTENTION_BRANCHES=1):
        super().__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = 1
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

    def forward(self, h):
        h = h.view(-1, self.feature_size)
        h = self.projector(h)  # [b x M]

        a = self.attention(h)  # [b x ATTENTION_BRANCHES]
        a = torch.transpose(a, 1, 0)  # [ATTENTION_BRANCHES x b]
        a = F.softmax(a, dim=1)  # softmax over b

        z = torch.mm(a, h)  # [ATTENTION_BRANCHES x M]
        return z, a

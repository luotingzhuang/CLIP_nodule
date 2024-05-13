import torch
from torch import nn
import config as CFG

from fmcib.models import LoadModel, fmcib_model
from transformers import DistilBertModel, DistilBertConfig

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


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, freeze=CFG.freeze):
        super().__init__()
        if pretrained:
            model = DistilBertModel.from_pretrained(model_name)
        else:
            model = DistilBertModel(config=DistilBertConfig())
            
        if CFG.freeze:
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                param.requires_grad = True

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]
    

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
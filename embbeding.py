
import torch
import torch.nn as nn
from torch.nn import functional as F


class FlexEmbedding(nn.Module):

    def __init__(self, categorical, continuous, sequence=0, embedding_dim=32):
        # categorical: number of categorical variables
        # continuous: number of continuous variables
        # sequence: number of sequence variables        
        super().__init__()
        self.categorical = categorical
        self.continuous = continuous
        self.sequence = sequence

        self.cat_emb = nn.Embedding(categorical, embedding_dim)
        self.cont_emb = nn.Linear(continuous, embedding_dim)
        self.seq_emb = nn.Linear(sequence, embedding_dim)

    def forward(self, categorical, continuous, sequence):
        # categorical: (batch_size, categorical)
        # continuous: (batch_size, continuous)
        # sequence: (batch_size, sequence)
        cat_emb = self.cat_emb(categorical)
        cont_emb = self.cont_emb(continuous)
        seq_emb = self.seq_emb(sequence)
        return cat_emb + cont_emb + seq_emb
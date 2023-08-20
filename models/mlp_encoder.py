import math
import torch
import torch.nn as nn
from functools import partial

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.gelu1 = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dense1(x)
        x = self.gelu1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x
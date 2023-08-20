import torch
import torch.nn as nn
import torch.nn.functional as F
from multiheadselfattention import MultiHeadSelfAttention
from masked_multiheadselfattention import MaskedMultiHeadSelfAttention
from mlp_encoder import MLP

# Define the model architecture
class Encoder(nn.Module):
    def __init__(self, d_model, nhead):
        super(Encoder, self).__init__()        

        self.self_attn = MultiHeadSelfAttention(d_model, num_heads=nhead)
        self.mlp = MLP(in_features, hidden_features=None, out_features=None)
        self.dense1 = nn.Linear(d_model, 512)
        self.dense2 = nn.Linear(512, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        V, K, Q = x[0], x[1], x[2]
        x = x + self.self_attn(V, K, Q)
        x = self.norm1(x)
        x = x+ self.mlp(x)
        output = self.norm2(x)

        return output
    

class Decoder(nn.Module):
    def __init__(self, d_model, nhead):
        super(Decoder, self).__init__()        

        self.masked_self_attn = MaskedMultiHeadSelfAttention(d_model, num_heads=nhead)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads=nhead)
        self.mlp = MLP(in_features, hidden_features=None, out_features=None)
        self.dense1 = nn.Linear(d_model, 512)
        self.dense2 = nn.Linear(512, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, V_e, k_e):
        V_d, K_d, Q_d = x[0], x[1], x[2]
        x1 = masked_self_attn(V_d, K_d, Q_d)
        x = x + x1
        x = self.norm1(x)

        Q_d = x[2]
        x = x + self.self_attn(V_d, K_d, Q_d)
        x = self.norm2(x)
        x = x + self.mlp(x)
        output = self.norm3(x)

        return output
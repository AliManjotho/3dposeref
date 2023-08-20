import math
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.relu1 = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout_atten = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.dropout_proj = nn.Dropout(0.2)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.dropout_atten(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout_proj(x)
        return x



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.dropout_path = DropPath(nn.Identity())
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.dropout_path(self.attn(self.norm1(x)))
        x = x + self.dropout_path(self.mlp(self.norm2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, h=8, length=76):
        super().__init__()

        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]  

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x += self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x





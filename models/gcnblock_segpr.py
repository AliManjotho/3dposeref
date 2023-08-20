import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNBlock_SEGPR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNBlock_SEGPR, self).__init__()
        
        self.graph_conv = GCNConv(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.max_pooling = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, x, edge_index):
        x = self.graph_conv(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm(x)
        x = self.max_pooling(x)
        return x




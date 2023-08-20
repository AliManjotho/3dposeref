import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNBlock_ASR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNBlock_ASR, self).__init__()
        
        self.graph_conv = GCNConv(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.max_pooling = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, x, edge_index):
        x = self.graph_conv(x, edge_index)
        x = F.relu(x)
        x = self.batch_norm(x)
        x = self.max_pooling(x)
        return x











class GraphConvolutionalStackedModel(nn.Module):
    def __init__(self):
        super(GraphConvolutionalStackedModel, self).__init__()
        
        self.graph_block1 = GCNBlock_ASR(in_channels=3, out_channels=64)
        self.graph_block2 = GCNBlock_ASR(in_channels=64, out_channels=128)
        self.graph_block3 = GCNBlock_ASR(in_channels=128, out_channels=256)
        
        self.max_pooling = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, x, edge_index):
        x = self.graph_block1(x, edge_index)
        x = self.max_pooling(x)
        
        x = self.graph_block2(x, edge_index)
        x = self.max_pooling(x)
        
        x = self.graph_block3(x, edge_index)
        x = self.max_pooling(x)
        
        return x

# Assuming you have your own data for the pose and edge indices
input_pose = torch.randn(batch_size, num_joints, 3)  # Replace with your actual data
edge_index = torch.tensor(...)  # Replace with your actual edge indices

model = GraphConvolutionalStackedModel()
output = model(input_pose, edge_index)
print(output.shape)  # Check the output shape

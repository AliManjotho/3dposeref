import torch
import torch.nn as nn
import torch.nn.functional as F
from gcnblock_asr import GCNBlock_ASR


class ASRModel(nn.Module):
    def __init__(self, num_joints):
        super(ASRModel, self).__init__()
        
        self.graph_block1 = GCNBlock_ASR(in_channels=3, out_channels=64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.graph_block2 = GCNBlock_ASR(in_channels=64, out_channels=128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.graph_block3 = GCNBlock_ASR(in_channels=128, out_channels=256)
        self.dropout3 = nn.Dropout(0.3)
        
        self.graph_block4 = GCNBlock_ASR(in_channels=256, out_channels=512)
        self.dropout4 = nn.Dropout(0.4)
        
        self.graph_block5 = GCNBlock_ASR(in_channels=512, out_channels=1024)
        
        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(512)
        
        self.dense2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 18)
        

    def forward(self, x, edge_index):
        # GraphConvolutionalBlock1
        x1 = self.graph_block1(x, edge_index)
        x1 = self.dropout1(x1)
        
        # GraphConvolutionalBlock2
        x2 = self.graph_block2(x1, edge_index)
        x2 = self.dropout2(x2)
        
        # Skip connection from Dropout layer 1 to GraphConvolutionalBlock3
        x3_input = x2 + x1
        
        # GraphConvolutionalBlock3
        x3 = self.graph_block3(x3_input, edge_index)
        x3 = self.dropout3(x3)
        
        # Skip connection from Dropout layer 3 to GraphConvolutionalBlock5
        x5_input = x3 + x2
        
        # GraphConvolutionalBlock4
        x4 = self.graph_block4(x3, edge_index)
        x4 = self.dropout4(x4)
        
        # GraphConvolutionalBlock5
        x5 = self.graph_block5(x5_input, edge_index)
        
        # Flatten
        x_flat = self.flatten(x5)
        
        # Fully connected layers
        x_fc1 = self.dense1(x_flat)
        x_fc1 = self.relu1(x_fc1)
        x_fc1 = self.batch_norm1(x_fc1)
        
        x_fc2 = self.dense2(x_fc1)
        x_fc2 = self.relu2(x_fc2)
        
        x_fc3 = self.dense3(x_fc2)        
        
        # Output layer
        output = self.dense4(x_fc3)
        
        return output

# Assuming you have your own data for the pose and edge indices
num_joints = 19  # Number of 3D joints
input_pose = torch.randn(batch_size, num_joints, 3)  # Replace with your actual data
edge_index = torch.tensor(...)  # Replace with your actual edge indices

model = ASRModel(num_joints)
output = model(input_pose, edge_index)
print(output.shape)  # Check the output shape

import torch
import torch.nn as nn
import torch.nn.functional as F
from gcnblock_segpr import GCNBlock_SEGPR
from regressionhead_segpr import RH1, RH2


class SEGPR(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SEGPR, self).__init__()

        self.gcn1 = GCNBlock_SEGPR(in_channels, hidden_channels)
        self.gcn2 = GCNBlock_SEGPR(hidden_channels, hidden_channels)
        self.gcn3 = GCNBlock_SEGPR(hidden_channels, hidden_channels)
        self.gcn4 = GCNBlock_SEGPR(hidden_channels, hidden_channels)
        self.gcn5 = GCNBlock_SEGPR(hidden_channels, hidden_channels)

        self.rh1 = RH1()
        self.rh2 = RH2()

    def forward(self, x, edge_index):
        skip_connections = []

        out1 = self.gcn1(x, edge_index)
        skip_connections.append(out1)

        out2 = self.gcn2(out1, edge_index)
        out3 = self.gcn3(out1 + out2, edge_index)
        skip_connections.append(out3)

        out4 = self.gcn4(out3, edge_index)
        out5 = self.gcn5(out3 + out4, edge_index)

        out6 = self.rh1(out5)
        out7 = self.rh2(out5)

        out8 = [out6, out7]

        return out8






# Assuming you have your own data for the pose and edge indices
num_joints = 19  # Number of 3D joints
input_pose = torch.randn(batch_size, num_joints, 3)  # Replace with your actual data
edge_index = torch.tensor(...)  # Replace with your actual edge indices

model = SEGPR(num_joints)
output = model(input_pose, edge_index)
print(output.shape)  # Check the output shape

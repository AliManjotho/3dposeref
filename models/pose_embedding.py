import torch
import torch.nn as nn
import torch.nn.functional as F

# Define embedding
class PoseEmbedding(nn.Module):
    def __init__(self, d_model=512):
        super(PoseEmbedding, self).__init__()
        self.dense1 = nn.Linear(3 * 19, d_model)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.dense1(x)
        output = self.relu1(x)

        return output

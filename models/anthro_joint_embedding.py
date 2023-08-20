import torch
import torch.nn as nn
import torch.nn.functional as F

# Define embedding
class AnthroJointEmbedding(nn.Module):
    def __init__(self, d_model=512):
        super(AnthroJointEmbedding, self).__init__()
        self.dense1 = nn.Linear((3 * 19) + 18, d_model)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.dense1(x)
        output = self.relu1(x)

        return output

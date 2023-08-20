import torch
import torch.nn as nn

class APRLoss(nn.Module):
    def __init__(self):
        super(APRLoss, self).__init__()

    def forward(self, predicted_joint, groundtruth_joint):
        squared_diff = (predicted_joint - groundtruth_joint) ** 2
        loss = torch.mean(torch.sum(squared_diff, dim=1))
        
        return loss
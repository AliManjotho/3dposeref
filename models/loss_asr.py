import torch
import torch.nn as nn

class ASRLoss(nn.Module):
    def __init__(self):
        super(ASRLoss, self).__init__()

    def forward(self, predicted_stature, groundtruth_stature):
        squared_diff = (predicted_stature - groundtruth_stature) ** 2
        loss = torch.mean(torch.sum(squared_diff, dim=1))
        
        return loss






# Example usage
batch_size = 32
num_bone_lengths = 18

# Generate some random data for demonstration
predicted_stature = torch.rand(batch_size, num_bone_lengths)
groundtruth_stature = torch.rand(batch_size, num_bone_lengths)

# Create an instance of the custom loss function
stature_loss = ASRLoss()

# Calculate the loss
loss_value = stature_loss(predicted_stature, groundtruth_stature)
print("Loss:", loss_value.item())

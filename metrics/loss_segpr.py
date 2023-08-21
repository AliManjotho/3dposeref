import torch
import torch.nn as nn

class RH1Loss(nn.Module):
    def __init__(self):
        super(RH1Loss, self).__init__()

    def forward(self, predicted_error, groundtruth_error):
        squared_diff = (predicted_error - groundtruth_error) ** 2
        loss = torch.mean(torch.sum(squared_diff, dim=1))
        
        return loss


class RH2Loss(nn.Module):
    def __init__(self):
        super(RH2Loss, self).__init__()

    def forward(self, predicted_inv_score, groundtruth_inv_score):
        squared_diff = (predicted_inv_score - groundtruth_inv_score) ** 2
        loss = torch.mean(torch.sum(squared_diff, dim=1))
        
        return loss



# Example usage
batch_size = 32
num_joint_comp= 19*3
num_inv_joints = 19-2

# Generate some random data for demonstration
predicted_comp= torch.rand(batch_size, num_joint_comp)
groundtruth_comp = torch.rand(batch_size, num_joint_comp)

predicted_inv_scores= torch.rand(batch_size, num_inv_joints)
groundtruth_inv_score = torch.rand(batch_size, num_inv_joints)

rh1_loss = RH1Loss()
rh2_loss = RH2Loss()

loss_rh1 = rh1_loss(predicted_comp, groundtruth_comp)
loss_rh2 = rh2_loss(predicted_inv_scores, groundtruth_inv_score)
print("RH1 Loss:", loss_rh1.item())
print("RH2 Loss:", loss_rh2.item())

import torch
import torch.nn as nn
import torch.nn.functional as F


class RH1(nn.Module):
    def __init__(self):
        super(RH1, self).__init__()
        
        self.dense1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(512)

        self.dense2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(512)

        self.dense3 = nn.Linear(128, 64)
        self.dense4 = nn.Linear(128, 57)
        

    def forward(self, x):

        x1 = self.dense1(x)
        x2 = self.relu1(x1)
        x3 = self.batch_norm1(x3)

        x4 = self.dense2(x3)
        x5 = self.relu2(x4)
        x6 = self.batch_norm2(x5)

        x7 = self.dense3(x6)

        # Output layer
        output = self.dense4(x7)
        
        return output
    



class RH2(nn.Module):
    def __init__(self):
        super(RH2, self).__init__()
        
        self.dense1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(512)

        self.dense2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(512)

        self.dense3 = nn.Linear(128, 64)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):

        x1 = self.dense1(x)
        x2 = self.relu1(x1)
        x3 = self.batch_norm1(x3)

        x4 = self.dense2(x3)
        x5 = self.relu2(x4)
        x6 = self.batch_norm2(x5)

        x7 = self.dense3(x6)

        # Output layer
        output = self.sigmoid(x7)
        
        return output












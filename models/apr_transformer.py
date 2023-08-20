import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncoding
from anthro_joint_embedding import AnthroJointEmbedding
from multiheadselfattention import MultiHeadSelfAttention
from pose_embedding import PoseEmbedding
from transformer import Encoder, Decoder


class APRTransformer(nn.Module):
    def __init__(self, num_encoders, num_decoders, d_model, nhead):
        super(APRTransformer, self).__init__()
        
        self.embedding_anthro = AnthroJointEmbedding(d_model)
        self.embedding_pose = PoseEmbedding(d_model)
        self.positional_encoding_encoder = PositionalEncoding(d_model)
        self.positional_encoding_decoder = PositionalEncoding(d_model)
        
        self.encoders = nn.ModuleList([Encoder(d_model, nhead) for _ in range(num_encoders)])
        self.decoders = nn.ModuleList([Decoder(d_model, nhead) for _ in range(num_decoders)])
        
        # Regression Head

        self.dense1 = nn.Linear(1024, 1024)
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.dense2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.dense3 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU()
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)
        
        self.dense4 = nn.Linear(d_model, 3 * 19)

    def forward(self, x):
        result = "<SOP>"
        x1 = self.embedding_anthro(x)
        x2 = self.positional_encoding_encoder(x)
        
        x = x1 + x2

        # Encoding
        for encoder in self.encoders:
            x = encoder(x)
        
        V_e, K_e, Q_e = x[0], x[1], x[2]

        y1 = self.embedding_pose(result)
        y2 = self.positional_encoding_decoder(result)

        y = y1 + y2

        V_d, K_d, Q_d = y[0], y[1], y[2]

        ys = []
        # Decoding
        for decoder in self.decoders:
            ys.append(decoder(y, V_e, K_e))

        y = torch.sum(ys, axis=1)

        y = self.dense1(y)
        y = self.relu1(y)
        y = self.batch_norm1(y)
        y = self.dropout1(y)

        y = self.dense2(y)
        y = self.relu2(y)
        y = self.batch_norm2(y)
        y = self.dropout2(y)

        y = self.dense3(y)
        y = self.relu3(y)
        y = self.batch_norm3(y)
        y = self.dropout3(y)
        
        output = self.dense4(y)

        return output



# Create an instance of the model
num_encoders = 6
num_decoders = 6
d_model = 512
nhead = 8

model = APRTransformer(num_encoders, num_decoders, d_model, nhead)
print(model)

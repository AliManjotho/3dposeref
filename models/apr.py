import torch
import torch.nn as nn
from einops import rearrange
from modules.encoder import Encoder
from model.module.trans_hypothesis import Transformer as Transformer_hypothesis

class APR(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.joints = 19
        self.joints_comps = 3
        self.anthro_bones = 18
        self.input_length = self.joints*self.joints_comps + self.anthro_bones
        self.output_length = 18

        ## Encoder blocks
        self.norm_1 = nn.LayerNorm(args.frames)
        self.norm_2 = nn.LayerNorm(args.frames)
        self.norm_3 = nn.LayerNorm(args.frames)
        self.norm_4 = nn.LayerNorm(args.frames)
        self.norm_5 = nn.LayerNorm(args.frames)
        self.norm_6 = nn.LayerNorm(args.frames)

        self.encoder1 = Encoder(4, args.frames, args.frames*2, length=self.input_length, h=8)
        self.encoder2 = Encoder(4, args.frames, args.frames*2, length=self.input_length, h=8)
        self.encoder3 = Encoder(4, args.frames, args.frames*2, length=self.input_length, h=8)
        self.encoder4 = Encoder(4, args.frames, args.frames*2, length=self.input_length, h=8)
        self.encoder5 = Encoder(4, args.frames, args.frames*2, length=self.input_length, h=8)
        self.encoder6 = Encoder(4, args.frames, args.frames*2, length=self.input_length, h=8)


        ## Embedding
        if args.frames > 27:
            self.embedding_1 = nn.Conv1d(self.input_length, args.channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(self.input_length, args.channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(self.input_length, args.channel, kernel_size=1)
            self.embedding_4 = nn.Conv1d(self.input_length, args.channel, kernel_size=1)
            self.embedding_5 = nn.Conv1d(self.input_length, args.channel, kernel_size=1)
            self.embedding_6 = nn.Conv1d(self.input_length, args.channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(self.length, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(self.length, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(self.length, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_4 = nn.Sequential(
                nn.Conv1d(self.length, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_5 = nn.Sequential(
                nn.Conv1d(self.length, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_6 = nn.Sequential(
                nn.Conv1d(self.length, args.channel, kernel_size=1),
                nn.BatchNorm1d(args.channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        ## SHR & CHI
        self.Transformer_hypothesis = Transformer_hypothesis(args.layers, args.channel, args.d_hid, length=args.frames)
        
        ## Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(args.channel*6, momentum=0.1),
            nn.Conv1d(args.channel*6, 6*self.output_length, kernel_size=1)
        )

    def forward(self, x):
        B, F, J, C = x.shape
        x = rearrange(x, 'b f j c -> b (j c) f').contiguous()

        ## MHG
        x1 = x  + self.encoder1(self.norm_1(x))
        x2 = x1 + self.encoder2(self.norm_2(x1)) 
        x3 = x2 + self.encoder3(self.norm_3(x2))
        
        ## Embedding
        x1 = self.embedding_1(x1).permute(0, 2, 1).contiguous() 
        x2 = self.embedding_2(x2).permute(0, 2, 1).contiguous()
        x3 = self.embedding_3(x3).permute(0, 2, 1).contiguous()

        ## SHR & CHI
        x = self.Transformer_hypothesis(x1, x2, x3) 

        ## Regression
        x = x.permute(0, 2, 1).contiguous() 
        x = self.regression(x) 
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x







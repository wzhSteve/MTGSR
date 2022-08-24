import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.SpectralInteraction import SpectralInteraction, SpectralInteractionLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np



class simple_linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(simple_linear, self).__init__()

        self.hidden_dim = (input_dim + output_dim) //2
        # decomp_window = 45
        self.conv = nn.Conv1d(input_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=True)
        self.linear1 = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.linear_trend1 = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.decomp1 = series_decomp(input_dim//2+1)

        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.linear_trend2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.decomp2 = series_decomp(self.hidden_dim//2+1)

        self.linear3 = nn.Linear(self.hidden_dim, output_dim, bias=False)
        self.linear_trend3 = nn.Linear(self.hidden_dim, output_dim, bias=False)
        self.decomp3 = series_decomp(self.hidden_dim//2+1)

        self.linear_out = nn.Linear(self.hidden_dim, output_dim, bias=False)
        self.conv_out = nn.Conv1d(in_channels=output_dim, out_channels=output_dim, kernel_size=1, bias=False)
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(7)
        self.linear_dim_in = nn.Linear(8, 4, bias=False)
        self.linear_dim_out = nn.Linear(4, 8, bias=False)

    def forward(self, input):
        # layer1_res, layer1_trend = self.decomp1(input)
        # layer1_res = self.linear1(layer1_res.permute(0, 2, 1)).permute(0, 2, 1)
        # layer1_trend = self.linear_trend1(layer1_trend.permute(0, 2, 1)).permute(0, 2, 1)
        #
        # # layer2_res, layer2_trend = self.decomp2(layer1_res)
        # # layer2_res = self.linear2(layer2_res.permute(0, 2, 1)).permute(0, 2, 1)
        # # layer2_trend = self.linear_trend2((layer2_trend + layer1_trend).permute(0, 2, 1)).permute(0, 2, 1)
        #
        # layer3_res, layer3_trend = self.decomp3(layer1_res)
        # layer3_res = self.linear3(layer3_res.permute(0, 2, 1)).permute(0, 2, 1)
        # layer3_trend = self.linear_trend3((layer1_trend + layer3_trend).permute(0, 2, 1)).permute(0, 2, 1)
        #
        # output = layer3_res + layer3_trend
        # output = self.linear_out(output.permute(0, 2, 1)).permute(0, 2, 1)
        #differential_learning

        #input = self.linear_dim_in(input)
        output = self.linear1(input.permute(0, 2, 1))
        # output = self.activation(output)
        output = self.linear_out(output).permute(0, 2, 1)
        # output = self.activation(output)
        #output = self.linear_dim_out(output)
        return output
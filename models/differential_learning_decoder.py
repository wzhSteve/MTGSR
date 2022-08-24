import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.SpectralInteraction import SpectralInteraction, SpectralInteractionLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np

from models.simple_linear import simple_linear

from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from scipy import signal

from layers.SelfAttention_Family import FullAttention, AttentionLayer
from models.differential_learning_encoder import differential_learning_encoder

class differential_decoder_layer(nn.Module):
    """
    功能：
    """
    def __init__(self, input_dim, output_dim, dropout=0.1, activation="gelu"):
        super(differential_decoder_layer, self).__init__()
        self.layer = simple_linear(input_dim, output_dim)
    def forward(self, layer_input):
        layer_output = self.layer(layer_input)
        return layer_output


class differential_learning_decoder(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim=7, dim_reduction=4, dropout=0.01, activation='gelu'):
        super(differential_learning_decoder, self).__init__()
        self.window_size = 12  # 第一层，二层，三层
        self.dim_reduction = dim_reduction
        self.layer = 2

        self.linear_combine_list = nn.ModuleList([])
        self.decomp_list = nn.ModuleList([])
        self.decoder_layer_list = nn.ModuleList([])
        self.differential_layer_list = nn.ModuleList([])
        self.decoder_layer_list2 = nn.ModuleList([])
        self.differential_layer_list2 = nn.ModuleList([])

        self.linear_out_list = nn.ModuleList([])
        self.conv_out_list = nn.ModuleList([])
        count = 0
        layer_dim = input_dim

        while (count < self.layer):
            layer_input_dim = self.window_size//self.dim_reduction
            layer_output_dim = self.window_size
            self.decomp_list.append(series_decomp(self.window_size // 2 + 1))
            # share the core layer
            self.linear_combine_list.append(differential_decoder_layer(2*layer_dim, layer_dim))
            self.decoder_layer_list.append(differential_decoder_layer(layer_input_dim, layer_output_dim))
            self.decoder_layer_list2.append(differential_decoder_layer(layer_input_dim, layer_output_dim))
            count = count + 1
            layer_dim = layer_dim * self.dim_reduction
            # #
            # self.linear_out_list.append(nn.Linear(layer_dim, layer_dim, bias=False))
            # self.conv_out_list.append(
            #     nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=5, stride=1, padding=2,
            #               padding_mode='circular', bias=False))

    def forward(self, input, layer_input):
        batch_size, sequence_len, feature_dim = input.shape
        count = 0
        layer_output = []  # 各层输出的list
        while (count < self.layer):
            #combine
            # input = torch.cat((input, layer_input[self.layer-1-count]), dim=1)
            # input = self.linear_combine_list[count](input)
            input = input + layer_input[self.layer-1-count]
            cnt = sequence_len // (self.window_size // self.dim_reduction)
            # 用于存储局部输出
            output = torch.tensor([]).to(input.device)
            for i in range(cnt):
                ii = i * (self.window_size//self.dim_reduction)
                # block index
                input_ii = input[:, ii:ii + self.window_size//self.dim_reduction, :]
                next_output = self.decoder_layer_list[count](input_ii)
                output = torch.cat((output, next_output), 1)  # 按sequenc_len这一维度拼接
            # print("encoder: 第{}次离散局部输出,output:[{},{},{}]".format(count,output.shape[0],output.shape[1],output.shape[2]))
            # output = self.linear_out_list[count]((output + output_div).permute(0, 2, 1)).permute(0, 2, 1)
            # output = self.conv_out_list[count](output.permute(0, 2, 1)).permute(0, 2, 1)
            input = output
            sequence_len = output.shape[1]
            count = count + 1  # 层数
        # output为最终隐藏层z ，layer_output为各层输出的list
        return output




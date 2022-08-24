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


class separate_encoder_layer(nn.Module):
    """
    功能：
    """
    def __init__(self, input_dim, output_dim, dropout=0.1, activation="gelu"):
        super(separate_encoder_layer, self).__init__()
        self.layer = simple_linear(input_dim, output_dim)
    def forward(self, layer_input):
        layer_output = self.layer(layer_input)
        return layer_output

class separate_linear(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.01, activation='gelu'):
        """
        :param step: 步长 即每个子序列的长度
        :param attn_layer: 注意力模块初始化
        """
        super(separate_linear, self).__init__()
        self.window_size = 12 #第一层，二层，三层
        self.layer = 1
        self.block_num = input_dim // self.window_size
        self.encoder_layer_list = nn.ModuleList([])
        count = 0
        while(count < self.layer):
            layer_input_dim = self.window_size
            layer_output_dim = output_dim // self.block_num
            # share the core layer
            self.encoder_layer_list.append(separate_encoder_layer(layer_input_dim, layer_output_dim))
            # don't share the core layer
            # temp = nn.ModuleList([])
            # for i in range(self.block_num):
            #     temp.append(separate_encoder_layer(layer_input_dim, layer_output_dim))
            # self.encoder_layer_list.append(temp)
            count = count + 1

        self.linear_out = nn.Linear(output_dim, output_dim, bias=False)

    def forward(self, input):
        batch_size, sequence_len, feature_dim = input.shape
        count = 0
        layer_output = [] #各层输出的list
        while(count < self.layer):
            #sequence由本层序列长度 cnt为本层分块数
            cnt = sequence_len//self.window_size
            #用于存储局部输出
            output = torch.tensor([]).to(input.device)
            for i in range(cnt):
                ii = i * self.window_size
                # block index
                input_ii = input[:, ii:ii + self.window_size, :]
                next_output = self.encoder_layer_list[count](input_ii)
                output = torch.cat((output, next_output), 1)  # 按sequenc_len这一维度拼接
            #print("encoder: 第{}次离散局部输出,output:[{},{},{}]".format(count,output.shape[0],output.shape[1],output.shape[2]))
            input = output
            sequence_len = output.shape[1]
            count = count + 1  # 层数
        #output为最终隐藏层z ，layer_output为各层输出的list
        output = self.linear_out(output.permute(0, 2, 1)).permute(0, 2, 1)
        return output



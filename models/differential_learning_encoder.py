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

class attention_layer(nn.Module):
    """
    功能：
    """
    def __init__(self,attention, d_model, dropout=0.1, activation="gelu", separate_factor=2, step=4):
        """
        :param attention: 注意力机制
        :param d_model:
        :param dff:
        :param dropout:
        :param activation:
        :param separate_factor: 为每层的输入与输出比
        :param step: 每层输入长度
        """
        super(attention_layer,self).__init__()
        self.step = step
        self.attention = attention #attentionlayer
        self.linear1 = nn.Linear(d_model, d_model,bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #降维的linear
        self.linear2 = nn.Linear(step, step//separate_factor,bias=True)
        self.norm3 = nn.LayerNorm(step//separate_factor)
        self.dropout = nn.Dropout(dropout)

        #self.activation = F.relu if activation == "relu" else F.gelu
        self.activation = F.elu
    def forward(self,x,attn_mask=None):
        """
        :param x: 输入
        :param attn_mask: mask
        :return:y=attn*value为attention模块的输出， attn就是softmax(qk^T)
        """
        new_x, attn = self.attention(x,x,x,attn_mask = attn_mask)
        y = x + self.dropout(new_x)
        #降维 y:[batch_size, step, d_model]->[batch_size, step//factor, d_model]
        y = self.dropout(self.activation(self.linear2(y.transpose(-1, -2))))
        y = y.transpose(-1, -2)

        return y, attn

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


class differential_learning_layer(nn.Module):
    """
    功能：
    """
    def __init__(self, input_dim, output_dim, dropout=0.1, activation="gelu"):
        super(differential_learning_layer, self).__init__()
        self.layer = simple_linear(input_dim, output_dim)
    def forward(self, l_input, r_input):
        differential_coef = l_input * r_input
        differential_coef = torch.softmax(torch.sum(differential_coef, dim=-1), dim=-1)
        differential_coef = differential_coef.unsqueeze(-1).repeat(1, 1, 7)
        differential_feature = l_input * differential_coef
        layer_output = self.layer(r_input + differential_feature)
        return layer_output

class differential_learning_encoder(nn.Module):
    def __init__(self, input_dim, output_dim, feature_dim=7, dim_reduction=4, dropout=0.01, activation='gelu'):
        """
        :param step: 步长 即每个子序列的长度
        :param attn_layer: 注意力模块初始化
        """
        super(differential_learning_encoder, self).__init__()
        self.activation = activation
        self.window_size = 12 #第一层，二层，三层
        self.dim_reduction = dim_reduction
        self.layer = 2
        self.block_num = input_dim // self.window_size
        self.decomp_list = nn.ModuleList([])
        self.encoder_layer_list = nn.ModuleList([])
        self.differential_layer_list = nn.ModuleList([])
        self.encoder_layer_list2 = nn.ModuleList([])
        self.differential_layer_list2 = nn.ModuleList([])

        self.linear_out_list = nn.ModuleList([])
        self.conv_out_list = nn.ModuleList([])
        self.attention_layer_list = nn.ModuleList([])
        self.attention_list = nn.ModuleList([])
        count = 0
        layer_dim = input_dim
        while(count < self.layer):
            layer_input_dim = self.window_size
            layer_output_dim = self.window_size//self.dim_reduction
            self.decomp_list.append(series_decomp(self.window_size//2+1))
            # share the core layer
            self.encoder_layer_list.append(separate_encoder_layer(layer_input_dim, layer_output_dim))
            self.differential_layer_list.append(differential_learning_layer(layer_input_dim, layer_output_dim))
            self.encoder_layer_list2.append(separate_encoder_layer(layer_input_dim, layer_output_dim))
            self.differential_layer_list2.append(differential_learning_layer(layer_input_dim, layer_output_dim))
            count = count + 1
            layer_dim = layer_dim//self.dim_reduction
            self.linear_out_list.append(nn.Linear(layer_dim, layer_dim, bias=False))
            self.conv_out_list.append(nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False))
            # self.attention_list.append(
            #     AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False), d_model,
            #                    n_heads, mix=mix))
            # self.attention_list.append(
            #     attention_layer(self.attention_layer_list[count], d_model, dropout=dropout,
            #                            activation=activation, step=self.step[count],
            #                            separate_factor=self.separate_factor[count]))

    def forward(self, input):
        batch_size, sequence_len, feature_dim = input.shape
        count = 0
        layer_output = [] #各层输出的list
        while(count < self.layer):
            #sequence由本层序列长度 cnt为本层分块数
            cnt = sequence_len//self.window_size
            #用于存储局部输出
            output = torch.tensor([]).to(input.device)
            output_div = torch.tensor([]).to(input.device)
            for i in range(cnt):
                ii = i * self.window_size
                # block index
                if i == 0:
                    input_ii = input[:, ii:ii + self.window_size, :]
                    temp_div, temp_mean = self.decomp_list[count](input_ii)
                    next_output = self.encoder_layer_list[count](temp_mean)
                    next_output_div = self.encoder_layer_list2[count](temp_div)
                    output = torch.cat((output, next_output), 1)  # 按sequenc_len这一维度拼接
                    output_div = torch.cat((output_div, next_output_div), 1)
                else:
                    l_input = input[:, ii-self.window_size:ii, :]
                    r_input = input[:, ii:ii + self.window_size, :]
                    l_div, l_mean = self.decomp_list[count](l_input)
                    r_div, r_mean = self.decomp_list[count](r_input)
                    next_output = self.differential_layer_list[count](l_mean, r_mean)
                    next_output_div = self.differential_layer_list2[count](l_div, r_div)
                    output = torch.cat((output, next_output), 1)  # 按sequenc_len这一维度拼接
                    output_div = torch.cat((output_div, next_output_div), 1)
            #print("encoder: 第{}次离散局部输出,output:[{},{},{}]".format(count,output.shape[0],output.shape[1],output.shape[2]))
            output = self.linear_out_list[count](output.permute(0, 2, 1)).permute(0, 2, 1)
            #
            output_div = self.linear_out_list[count](output_div.permute(0, 2, 1)).permute(0, 2, 1)
            # output = self.conv_out_list[count](output.permute(0, 2, 1)).permute(0, 2, 1)
            input = output
            layer_output.append(output_div)
            sequence_len = output.shape[1]
            count = count + 1  # 层数

        #output为最终隐藏层z ，layer_output为各层输出的list
        return output, layer_output



import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os
from models.simple_linear import simple_linear
from layers.distribution_block import distribution_block
from torch.nn.parameter import Parameter


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.einsum("bsl,le->bse", inputs, self.weight)
        output = torch.einsum("bsl,ble->bse", adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class TimeDimGraphModel(nn.Module):

    def __init__(self, configs):
        super(TimeDimGraphModel, self).__init__()
        self.configs = configs
        self.head_num = configs.head_num
        self.TimeDimGraphLayer_list = nn.ModuleList([])
        self.linear_list = nn.ModuleList([])
        for i in range(self.head_num):
            self.linear_list.append(nn.Linear(configs.c_out, configs.d_model))
            self.TimeDimGraphLayer_list.append(TimeDimGraphLayer(configs.seq_len, configs.pred_len, configs.d_model, configs.dropout, configs.layer_num, configs.block_size, configs.cover_size))
        self.projection = nn.Linear(self.head_num*configs.d_model, configs.c_out)
        self.dropout = nn.Dropout(configs.dropout)
        # self.contrast_loss = Contrast()

    def forward(self, input):
        B, L, D = input.shape
        output_list = torch.tensor([]).to(input.device)
        temp_input_list = []
        for i in range(self.head_num):
            temp_input = self.dropout(F.relu(self.linear_list[i](input)))
            temp_input_list.append(temp_input)
            temp_output = self.TimeDimGraphLayer_list[i](temp_input)
            output_list = torch.cat([output_list, temp_output], dim=-1)
        output = self.dropout(self.projection(output_list))
        return output


class TimeDimGraphLayer(nn.Module):

    def __init__(self, input_length, pred_length, dim, dropout, layer_num=1, block_size=24, cover_size=12):
        super(TimeDimGraphLayer, self).__init__()
        self.input_length = input_length
        self.dim = dim
        self.layer_num = layer_num
        self.block_size = block_size
        self.cover_size = cover_size
        self.dropout = nn.Dropout(dropout)

        self.pred_length = pred_length

        self.TimeDimGraphBlock = TimeDimGraphBlock(input_length=self.input_length, dim=dim, dropout=dropout, layer_num=layer_num, block_size=block_size, cover_size=cover_size)
        self.output_length = block_size * ((self.input_length - block_size) // (block_size - cover_size) + 1 - layer_num)
        # self.projection = nn.Conv1d(in_channels=self.output_length, out_channels=pred_length, kernel_size=5, stride=1, padding=2,
        #                             padding_mode='circular', bias=False)
        self.simple_linear = simple_linear(input_dim=self.output_length, output_dim=pred_length)
        self.linear = nn.Linear(self.output_length, pred_length)
        self.seasonal_linear = nn.Linear(self.output_length, pred_length)
        self.trend_linear = nn.Linear(self.output_length, pred_length)

    def forward(self, input):
        B, L, D = input.shape
        temp_output= self.TimeDimGraphBlock(input)
        output = self.simple_linear(temp_output)
        return output


class TimeDimGraphBlock(nn.Module):

    def __init__(self, input_length, dim, dropout, layer_num, block_size=24, cover_size=12):
        super(TimeDimGraphBlock, self).__init__()
        self.input_length = input_length
        self.dim = dim
        self.layer_num = layer_num
        self.block_size = block_size
        self.cover_size = cover_size
        self.residue_size = block_size - cover_size
        self.block_num = (input_length - block_size) // self.residue_size

        self.TimeGenerateGraph_module = nn.ModuleList([])
        self.DimGenerateGraph_module = nn.ModuleList([])
        self.CrossGenerateGraph_module = nn.ModuleList([])
        self.GCN_Time = nn.ModuleList([])
        self.GCN_Dim = nn.ModuleList([])
        self.GCN_Block = nn.ModuleList([])
        self.projection_list = nn.ModuleList([])
        for l in range(layer_num):
            self.TimeGenerateGraph_module.append(TimeGenerateGraph(block_size, dim, dropout))
            self.DimGenerateGraph_module.append(DimGenerateGraph(block_size, dim, dropout))
            self.CrossGenerateGraph_module.append(CrossGenerateGraph(block_size, dim, dropout))
            self.GCN_Time.append(GraphConvolution(dim, dim))
            self.GCN_Dim.append(GraphConvolution(block_size, block_size))
            self.GCN_Block.append(GraphConvolution(dim, dim))
            if l == 0:
                self.projection_list.append(
                    nn.Linear(self.input_length, block_size * ((self.input_length - block_size) // (block_size - cover_size) - l)))
            else:
                self.projection_list.append(nn.Linear(block_size * ((self.input_length - block_size) // (block_size - cover_size) + 1 - l), block_size * ((self.input_length - block_size) // (block_size - cover_size) - l)))

        self.dropout_Time = nn.Dropout()
        self.dropout_Dim = nn.Dropout()
        self.decomp = series_decomp(1 + block_size)

    def forward(self, input):
        B, L, D = input.shape
        for l in range(self.layer_num):
            output = torch.tensor([]).to(input.device)
            block_num = self.block_num - l
            for i in range(block_num):
                if l == 0:
                    ii = i * self.residue_size
                    temp_input1 = input[:, ii:ii + self.block_size, :]
                    temp_input2 = input[:, ii+ self.residue_size:ii + self.residue_size + self.block_size, :]
                else:
                    ii = i * self.block_size
                    temp_input1 = input[:, ii:ii + self.block_size, :]
                    temp_input2 = input[:, ii + self.block_size:ii + 2 * self.block_size, :]

                TimeGraph = self.TimeGenerateGraph_module[l](temp_input1) # B, block_size, block_size
                DimGraph = self.DimGenerateGraph_module[l](temp_input1) # B, D, D
                BlockGraph = self.CrossGenerateGraph_module[l](temp_input1, temp_input2) # B, block_size, block_size

                BlockVector = self.GCN_Block[l](temp_input1, BlockGraph)  # B, block_size, D
                BlockVector = self.dropout_Time(F.relu(BlockVector)) + temp_input1

                TimeDimVector = self.GCN_Dim[l](BlockVector.permute(0, 2, 1), DimGraph)
                TimeDimVector = self.dropout_Dim(F.relu(TimeDimVector))
                TimeDimVector = TimeDimVector.permute(0, 2, 1) + BlockVector

                TimeDimBlockVector = self.GCN_Time[l](TimeDimVector, TimeGraph)
                TimeDimBlockVector = self.dropout_Time(F.relu(TimeDimBlockVector)) + TimeDimVector

                output = torch.cat([output, TimeDimBlockVector], dim=1)
            # trend_temp, seasonal_temp = self.decomp(trend_output)
            # seasonal_output = self.projection_list[l](seasonal.permute(0, 2, 1)).permute(0, 2, 1) + seasonal_temp
            input = output

        return output # B, L - block_size, D


class TimeGenerateGraph(nn.Module):

    def __init__(self, input_length, dim, dropout):
        super(TimeGenerateGraph, self).__init__()
        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        # self.GCN = GraphConvolution(dim, dim)

    def forward(self, input):

        B, L, D = input.shape
        input = self.projection(input.permute(0, 2, 1)).transpose(1, 2)

        mean_input = input.mean(1, keepdim=True)
        std_input = torch.sqrt(torch.var(input, dim=1, keepdim=True, unbiased=False) + 1e-5)

        input = (input - mean_input.repeat(1, L, 1)) / std_input
        # scale为None则为1./sqrt(E) 为true即有值，则为该值
        scale = 1. / sqrt(D)
        # 内积 scores bhll
        scores = torch.einsum("ble,bse->bls", input, input)
        cross_value = self.dropout(F.softmax((scale * scores), -3, _stacklevel=5))

        # # GCN
        # TimeDimVector = self.GCN(input, cross_value)
        return cross_value # B, L, L


class DimGenerateGraph(nn.Module):

    def __init__(self, input_length, dim, dropout):
        super(DimGenerateGraph, self).__init__()
        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=input_length-1, out_channels=input_length-1, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        # self.GCN = GraphConvolution(dim, dim)

    def forward(self, input):
        input = torch.diff(input, dim=1)
        B, L, D = input.shape
        input = self.projection(input).permute(0, 2, 1) # B, D, L

        mean_input = input.mean(1, keepdim=True)
        std_input = torch.sqrt(torch.var(input, dim=1, keepdim=True, unbiased=False) + 1e-5)

        input = (input - mean_input.repeat(1, D, 1)) / std_input
        scale = 1. / sqrt(L)
        # 内积 scores bhll
        scores = torch.einsum("ble,bse->bls", input, input)
        cross_value = self.dropout(F.softmax((scale * scores), -3, _stacklevel=5)) # B, D, D
        # # GCN
        # TimeDimVector = self.GCN(input, cross_value)
        return cross_value # B, D, D


class CrossGenerateGraph(nn.Module):

    def __init__(self, input_length, dim, dropout):
        super(CrossGenerateGraph, self).__init__()
        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout(dropout)
        self.projection1 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        self.projection2 = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1,
                                    padding=2,
                                    padding_mode='circular', bias=False)
        # self.GCN = GraphConvolution(dim, dim)

    def forward(self, input1, input2):

        B, L, D = input1.shape
        input1 = self.projection1(input1.permute(0, 2, 1)).permute(0, 2, 1) # B, L, D
        input2 = self.projection2(input2.permute(0, 2, 1)).permute(0, 2, 1)  # B, L, D

        mean_input1 = input1.mean(1, keepdim=True)
        std_input1 = torch.sqrt(torch.var(input1, dim=1, keepdim=True, unbiased=False) + 1e-5)
        mean_input2 = input2.mean(1, keepdim=True)
        std_input2 = torch.sqrt(torch.var(input2, dim=1, keepdim=True, unbiased=False) + 1e-5)

        input1 = (input1 - mean_input1.repeat(1, L, 1)) / std_input1
        input2 = (input2 - mean_input2.repeat(1, L, 1)) / std_input2

        # scale为None则为1./sqrt(E) 为true即有值，则为该值
        scale = 1. / sqrt(D)
        # 内积 scores bhll
        scores = torch.einsum("ble,bse->bls", input1, input2)
        cross_value = self.dropout(F.softmax((scale * scores), -3, _stacklevel=5))
        # # GCN
        # TimeDimVector = self.GCN(input, cross_value)
        return cross_value  # B, L, L



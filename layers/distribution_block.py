import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from models.simple_linear import simple_linear
import math
import numpy as np
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class distribution_block(nn.Module):
    def __init__(self, seq_len, label_len, pred_len, feature_dim, window_size=24, dropout=0.01, activation='gelu', d_model=512):
        super(distribution_block, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.window_size = window_size
        # seq_len
        self.seq_num = seq_len // self.window_size
        # count = 0
        # while(count < self.seq_num):
        #     count = count + 1
        # label_len
        self.label_num = label_len // self.window_size
        # pred_len
        self.pred_num = pred_len // self.window_size

        self.conv_mean = nn.Conv1d(in_channels=self.seq_num, out_channels=self.pred_num, kernel_size=3, padding=1, bias=False)
        self.conv_std = nn.Conv1d(in_channels=self.seq_num, out_channels=self.pred_num, kernel_size=3, padding=1, bias=False)
        self.activation = F.elu
        self.trade_off1 = nn.Parameter(torch.ones(1, 1, feature_dim))
        self.trade_off2 = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.dropout = nn.Dropout(dropout)
        self.attention_layer = AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=False), feature_dim, 1)

    def forward(self, seq, label_pred):
        batch_size, seq_len, d_model = seq.shape
        # seq_len
        seq_mean = torch.tensor([]).to(seq.device)
        seq_std = torch.tensor([]).to(seq.device)
        seq_out = torch.tensor([]).to(seq.device)
        for i in range(self.seq_num):
            ii = i * self.window_size
            # 更新x_mean, x_div
            seq_ii = seq[:, ii:ii + self.window_size, :]
            temp_mean = seq_ii.mean(1, keepdim=True)
            temp_std = torch.sqrt(torch.var(seq_ii, dim=1, keepdim=True, unbiased=False) + 1e-5)
            seq_ii = (seq_ii - temp_mean) / temp_std

            seq_out = torch.cat((seq_out, seq_ii), 1)  # 按sequenc_len这一维度拼接
            seq_mean = torch.cat((seq_mean, temp_mean), 1)
            seq_std = torch.cat((seq_std, temp_std), 1)
        # label_pred_len
        label_pred_mean = torch.tensor([]).to(label_pred.device)
        label_pred_std = torch.tensor([]).to(label_pred.device)
        label_pred_out = torch.tensor([]).to(label_pred.device)

        for i in range(self.pred_num + self.label_num):
            ii = i * self.window_size
            # 更新x_mean, x_div
            label_pred_ii = label_pred[:, ii:ii + self.window_size, :]
            temp_mean = label_pred_ii.mean(1, keepdim=True)
            temp_std = torch.sqrt(torch.var(label_pred_ii, dim=1, keepdim=True, unbiased=False) + 1e-5)
            label_pred_ii = (label_pred_ii - temp_mean) / temp_std

            label_pred_out = torch.cat((label_pred_out, label_pred_ii), 1)  # 按sequenc_len这一维度拼接
            label_pred_mean = torch.cat((label_pred_mean, temp_mean), 1)
            label_pred_std = torch.cat((label_pred_std, temp_std), 1)

        seq_mean, _ = self.attention_layer(seq_mean, seq_mean, seq_mean, attn_mask=None)
        seq_std, _ = self.attention_layer(seq_std, seq_std, seq_std, attn_mask=None)

        pred_mean = self.conv_mean(self.dropout(self.activation(seq_mean)))
        pred_std = self.conv_std(self.dropout(self.activation(seq_std)))

        label_pred_mean = torch.cat((label_pred_mean[:, :self.label_num, :], pred_mean), 1)
        label_pred_std = torch.cat((label_pred_std[:, :self.label_num, :], pred_std), 1)

        label_pred_mean = self.trade_off1 * label_pred_mean.unsqueeze(2).repeat(1, 1, self.window_size, 1)
        label_pred_mean = label_pred_mean.view(batch_size, -1, d_model)

        label_pred_std = self.trade_off2 * label_pred_std.unsqueeze(2).repeat(1, 1, self.window_size, 1)
        label_pred_std = label_pred_std.view(batch_size, -1, d_model)

        return seq_out, label_pred_out, label_pred_mean, label_pred_std, pred_mean, pred_std

def distribution_loss(input, window_size=24):
    B, L, D = input.shape
    # seq_len
    input_mean = torch.tensor([]).to(input.device)
    input_std = torch.tensor([]).to(input.device)
    input_num = L // window_size
    for i in range(input_num):
        ii = i * window_size
        # 更新x_mean, x_div
        input_ii = input[:, ii:ii + window_size, :]
        temp_mean = input_ii.mean(1, keepdim=True)
        temp_std = torch.sqrt(torch.var(input_ii, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input_mean = torch.cat((input_mean, temp_mean), 1)
        input_std = torch.cat((input_std, temp_std), 1)
    return input_mean, input_std

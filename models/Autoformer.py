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
from models.separate_linear import separate_linear
from models.differential_learning import differential_learning
from models.differential_learning_encoder import differential_learning_encoder
from models.differential_learning_decoder import differential_learning_decoder
from layers.MultiGraph import MultiGraph, TimeDimGraphModel

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        self.simple_layer = simple_linear(input_dim = self.seq_len, output_dim = self.pred_len)
        self.separate_linear = separate_linear(input_dim = self.seq_len, output_dim = self.pred_len)
        self.differential_learning = differential_learning(input_dim = self.seq_len, output_dim = self.pred_len)
        self.differential_learning_encoder = differential_learning_encoder(input_dim=self.seq_len, output_dim=self.pred_len)
        self.differential_learning_decoder = differential_learning_decoder(input_dim=21,
                                                                           output_dim=self.pred_len)
        self.simple_linear = simple_linear(input_dim=21, output_dim=self.pred_len)
        self.MultiGraph = MultiGraph(configs.c_out, d_model=32, n_heads=4, seq_len=self.seq_len, pred_len=self.pred_len, dropout=configs.dropout)
        ### RIN Parameters ###
        self.RIN = True
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.c_out))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.c_out))

        self.trade_off1 = nn.Parameter(torch.ones(1, 1, configs.c_out))
        self.trade_off2 = nn.Parameter(torch.zeros(1, 1, configs.c_out))
        self.trend_graph = TimeDimGraphModel(configs)
        self.seasonal_graph = TimeDimGraphModel(configs)
        self.decompose = series_decomp(25)
        self.seasonal_linear = simple_linear(input_dim=self.seq_len, output_dim=self.pred_len)
        self.trend_linear = simple_linear(input_dim=self.seq_len, output_dim=self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        if self.RIN:
            print('/// RIN ACTIVATED ///\r', end='')
            means = x_enc.mean(1, keepdim=True).detach()
            #mean
            x_enc = x_enc - means
            #var
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x_enc = x_enc * self.affine_weight + self.affine_bias

        # output1 = self.simple_layer(x_enc)
        # output2 = self.MultiGraph(x_enc)
        # x_enc_res, x_enc_trend = self.decompose(x_enc)
        # output = self.trend_graph(x_enc_trend) + self.seasonal_linear(x_enc_res)

        output = self.trend_graph(x_enc)
        # output = output1 + self.trade_off2 * output3

        # output = self.separate_linear(x_enc)

        # input, layer_input = self.differential_learning_encoder(x_enc)
        # output = self.differential_learning_decoder(input, layer_input)
        # output = self.differential_learning(x_enc)

        ### reverse RIN ###
        if self.RIN:
            output = output - self.affine_bias
            output = output / (self.affine_weight + 1e-10)
            output = output * stdev
            output = output + means

        return output
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
# from layers.SpectralInteraction import SpectralInteraction, SpectralInteractionLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np
from layers.MultiGraph import TimeDimGraphModel

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
        ### RIN Parameters ###
        self.RIN = True
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.c_out))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.c_out))
        #     
        self.MTGSR = TimeDimGraphModel(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: [batch, seq_len, c_in]

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

        output = self.MTGSR(x_enc) # output [batch, pred_len, c_in]

        ### reverse RIN ###
        if self.RIN:
            output = output - self.affine_bias
            output = output / (self.affine_weight + 1e-10)
            output = output * stdev
            output = output + means

        return output
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


class SpectralInteraction(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpectralInteraction, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    #cos sim
    def sim(self, z1, z2):
        """
        input: z1 B,H,L,D
        input: z1 B,H,L,D
        output: sim_matrix B,H,L,L
        """
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.matmul(z1, z2.permute(0, 1, 3, 2).contiguous())
        dot_denominator = torch.matmul(z1_norm, z2_norm.permute(0, 1, 3, 2).contiguous())
        sim_matrix = torch.exp(dot_numerator / dot_denominator)
        return sim_matrix

    def forward(self, queries, keys, values, attn_mask):
        # queries.shape = values.shape
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]
        # cos sim ->B,H,L,L
        interaction = self.sim(queries.permute(0, 2, 1, 3).contiguous(), keys.permute(0, 2, 1, 3).contiguous())
        # interaction fft B,H,L,L-> B,H,L,L/2+1
        interaction_fft = torch.fft.rfft(interaction, dim=-1).real
        # value fft  B,H,L,E-> B,H,E,L/2+1
        v_fft = torch.fft.rfft(values.permute(0, 2, 3, 1).contiguous(), dim=-1)
        # scale为None则为1./sqrt(E) 为true即有值，则为该值
        scale = self.scale or 1. / sqrt(E)
        # A-> B,H,1,L/2+1
        A = self.dropout(torch.softmax(torch.sum(torch.softmax(scale * interaction_fft, dim=-1), dim=-2), dim=-1)).view(B, H, 1, -1)
        # v_real-> B,H,E,L/2+1
        v_real = v_fft.real
        # V_fft->B,H,L,E
        V_fft_real = A * v_real
        # # V ifft V->B,H,E,L
        V_complex = torch.complex(V_fft_real, v_fft.imag)
        V = torch.fft.irfft(V_complex, dim=-1)
        if self.output_attention:
            return (V.permute(0, 3, 2, 1).contiguous(), A)
        else:
            return (V.permute(0, 3, 2, 1).contiguous(), None)


class SpectralInteractionLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(SpectralInteractionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        #L=S D
        B, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # linear mapping
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        #spectral interaction
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


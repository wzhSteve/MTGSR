import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


def correlation_compute_H(self, x1, x2):

    B, H, L, E = x1.shape
    _, _, S, D = x2.shape

    mean_x1 = x1.mean(1, keepdim=True)
    mean_x2 = x2.mean(1, keepdim=True)
    std_x1 = x1.sqrt(1, keepdim=True) + 1e-5
    std_x2 = x2.sqrt(1, keepdim=True) + 1e-5

    x1 = (x1 - mean_x1.repeat(1, 1, L, 1)) / std_x1
    x2 = (x2 - mean_x2.repeat(1, 1, S, 1)) / std_x2
    # scale为None则为1./sqrt(E) 为true即有值，则为该值
    scale = self.scale or 1. / sqrt(E)
    # 内积 scores bhll
    scores = torch.einsum("bhle,bhse->bhls", x1, x2)
    softmax = nn.Softmax2d()
    cross_value = self.dropout(softmax(scale * scores))

    return cross_value.contiguous()

def correlation_compute(self, x1, x2):

    B, L, E = x1.shape
    _, S, D = x2.shape

    mean_x1 = x1.mean(1, keepdim=True)
    mean_x2 = x2.mean(1, keepdim=True)
    std_x1 = x1.sqrt(1, keepdim=True) + 1e-5
    std_x2 = x2.sqrt(1, keepdim=True) + 1e-5

    x1 = (x1 - mean_x1.repeat(1, L, 1)) / std_x1
    x2 = (x2 - mean_x2.repeat(1, S, 1)) / std_x2
    # scale为None则为1./sqrt(E) 为true即有值，则为该值
    scale = self.scale or 1. / sqrt(E)
    # 内积 scores bhll
    scores = torch.einsum("ble,bse->bls", x1, x2)
    softmax = nn.Softmax2d()
    cross_value = self.dropout(softmax(scale * scores))

    return cross_value.contiguous()


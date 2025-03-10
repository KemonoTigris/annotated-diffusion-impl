import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F


# Network helpers
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# This function increases the spatial resolution (height and width) of the input tensor
def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"), # Doubles spatial dimensions
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1), # Convolutional layer to refine upsampled output
    )

def Downsample(dim, dim_out=None):
    return nn.Sequential(
         # Reduces spatial dimensions by unshuffling pixels
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # 1x1 convolution to adjust channel count
        nn.Conv2d(dim * 4, default(dim_out, dim), 1), 
    )

# Position embeddings are a way for neural network to know which time-step (noise level)
# it's operating at.
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # The literal device - CPU/GPU - extracted to ensure that computations happen on the same device
        device = time.device
        # Divides embedding dimentionality into two halves for sine and cosine components
        half_dim = self.dim // 2

        scaling_factor = math.log(10000)/ (half_dim - 1)

        # 1D tensor of frequencies
        frequencies = torch.exp(torch.arange(half_dim, device=device) * scaling_factor)

        embeddings = time[:, None] * frequencies[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim = -1)
        return embeddings


# ResNet block

# 2D convolutional layer
# Pictures have RGB channels, so images can be represented like [channels, height, width]
# Pytorch can process batches of images, so image data can look like:
# [batch_size, channels, height, width]
class WeightStandardizedConv2d(nn.Conv2d):

    # x is typically a 4D tensor in shape [batch_size, channels, height, width]
    def forward(self, x):

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        # tensor of learnable parameters (the convolution filters) in the parent class
        # typically has shape [out_channels, in_channels, kernel_height, kernel_width]
        #  out_channels: how many different filters the convolution layer has
        #  in_channels: how many channels it expects as input (often 3 for RGB, possibly more)
        # We're grabing a reference to these filters to modify/"standardize" them
        weight = self.weight
    

        # changing the shape from [out_channels, in_chnnels, kernel_height, kernel_width] to [out_channels, 1, 1, 1]
        # takes the average value from all the other dimensions
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        # calculate the variance, similar to how we computed the mean above
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))

        # weight - mean gives us a weight tensor with an aveage of 0 per filter
        # (var + eps).rsqrt() is 1 / sqrt(var + eps)
        # eps is non-zero to avoid dividing by 0
        # each filter in normalized_weight has mean 0 and variance 1 
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        # Perform 2D convolution
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

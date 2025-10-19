import torch
import torch.nn as nn
from typing import List

class Downsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)
        
    def forward(self, x):
        return self.conv(x)
    
class Conv2dBlock(nn.Module):
    '''
        Conv2d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels height width -> batch channels 1 height width'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 height width -> batch channels height width'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
class Conv2dTransposeBlock(nn.Module):
    '''
        Conv2d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels height width -> batch channels 1 height width'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 height width -> batch channels height width'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class Conv2dUpsampler(nn.Module):
    """
        A simple Conv2d upsampler using ConvTranspose2d
        This can be used for decoding images from a lower resolution feature map
    """
    
    def __init__(self, channels: List[int], kernel_size: int):
        super().__init__()
        self.input_dim = channels[0]
        self.layers = []
        for i in range(len(channels) - 1):
            self.layers.append(Conv2dTransposeBlock(channels[i], channels[i + 1], kernel_size))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
    
class MLP(nn.Module):
    """
        A simple multi-layer perceptron
        This can be used for decoding actions from a flattened feature map
    """
    
    def __init__(self, channels: List[int], dropout=0.1):
        super().__init__()
        self.layers = []
        for i in range(len(channels) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(channels[i], channels[i + 1]),
                    nn.Mish(),
                    nn.Dropout(dropout)
                )
            )
        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.layers(x)


def test():
    cb = Conv2dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1,256,16,16))
    o = cb(x)

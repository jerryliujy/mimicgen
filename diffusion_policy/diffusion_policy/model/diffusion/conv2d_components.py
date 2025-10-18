import torch
import torch.nn as nn

def make_mlp(dimensions, activation=nn.Mish, dropout=0.1):
    layers = []
    for i in range(len(dimensions) - 1):
        layers.append(nn.Linear(dimensions[i], dimensions[i+1]))
        if i < len(dimensions) - 2:
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

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


def test():
    cb = Conv2dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1,256,16,16))
    o = cb(x)

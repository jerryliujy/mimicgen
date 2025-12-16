
"""
Adjust the code from https://github.com/jayLEE0301/vq_bet_official/blob/main/vector_quantize_pytorch/vector_quantize_pytorch.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
import einops
import numpy as np

from diffusion_policy.model.diffusion.conv1d_components import (
    Conv1dBlock, Downsample1d, Upsample1d
)
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        # For Conv1d, weight shape is (out_channels, in_channels, kernel_size)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid], gain)
        

class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        down_dims,
        dropout=0.2
    ):
        super(EncoderMLP, self).__init__()

        if len(down_dims) == 0:
            self.encoder = nn.Linear(input_dim, output_dim)
            self.fc = nn.Identity()
        else:
            layers = []
            layers.append(nn.Linear(input_dim, down_dims[0]))
            layers.append(nn.ReLU())
            for i in range(1, len(down_dims)):
                layers.append(nn.Linear(down_dims[i-1], down_dims[i]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            self.encoder = nn.Sequential(*layers)
            self.fc = nn.Linear(down_dims[-1], output_dim)
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        return state


class Encoder1DCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        down_dims,
        kernel_size=3,
        dropout=0.2
    ):
        super(Encoder1DCNN, self).__init__()
        blocks = list()
        current_dim = input_dim
        if len(down_dims) == 0:
            self.encoder = nn.Conv1d(input_dim, output_dim, 1)
            self.fc = nn.Identity()
        else:
            for i, dim in enumerate(down_dims):
                blocks.append(Conv1dBlock(current_dim, dim, kernel_size=kernel_size))
                blocks.append(Downsample1d(dim))
                blocks.append(nn.Dropout(dropout))
                current_dim = dim
            self.encoder = nn.Sequential(*blocks)
            self.fc = nn.Conv1d(down_dims[-1], output_dim, 1) 
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.avg_pool(self.fc(h))
        return state

        
class Decoder1DCNN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 up_dims, 
                 kernel_size=3,
                 dropout=0.2,
                 T=8
        ):
        super().__init__()
        blocks = list()
        current_dim = input_dim
        self.T = T
        if len(up_dims) == 0:
            self.decoder = nn.Identity()
            self.fc = nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1)
        else:
            for i, dim in enumerate(up_dims):
                blocks.append(Conv1dBlock(current_dim, dim, kernel_size=kernel_size))
                blocks.append(Upsample1d(dim))
                blocks.append(nn.Dropout(dropout))
                current_dim = dim
            self.decoder = nn.Sequential(*blocks)
            self.fc = nn.Conv1d(current_dim, output_dim, kernel_size=3, padding=1)
        self.apply(weights_init_encoder)

    def forward(self, x):
        # x: (B, input_dim, T_latent)
        x = einops.repeat(x, 'B D 1 -> B D T', T=self.T)
        h = self.decoder(x)
        x = self.fc(h)
        return x


class ActionAe(ModuleAttrMixin):
    def __init__(
        self,
        input_dim_h=16,  # length of action chunk
        input_dim_w=7,  # action dim
        down_dims=[128, 256],
        n_latent_dims=512,
        act_scale=1.0,
        use_mlp=True,  
        dropout=0.2
    ):
        super(ActionAe, self).__init__()
        self.n_latent_dims = n_latent_dims
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.act_scale = act_scale
        self.use_mlp = use_mlp
        
        if down_dims is None:
            down_dims = []
            up_dims = []
        else:
            up_dims = list(reversed(down_dims))

        if use_mlp:
            input_dim_w = input_dim_h * input_dim_w
            self.encoder = EncoderMLP(
                input_dim=input_dim_w, output_dim=n_latent_dims, 
                down_dims=down_dims,
                dropout=dropout
            )
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims, output_dim=input_dim_w, 
                down_dims=up_dims,
                dropout=dropout
            )
        else:
            self.encoder = Encoder1DCNN(
                input_dim=input_dim_w,
                output_dim=n_latent_dims,
                down_dims=down_dims,
                dropout=dropout
            )
            self.decoder = Decoder1DCNN(
                input_dim=n_latent_dims,
                output_dim=input_dim_w,
                up_dims=up_dims,
                dropout=dropout,
                T=input_dim_h // (len(down_dims) + 1)
            )
            

    def get_action_from_latent(self, latent):
        if self.use_mlp:
            output = self.decoder(latent) * self.act_scale
            # (B, T * A) -> (B, T, A)
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            output = self.decoder(latent) * self.act_scale
            # (B, A, T) -> (B, T, A)
            return output.permute(0, 2, 1)


    def preprocess(self, state):
        if self.use_mlp:
            state = einops.rearrange(state, "N T A -> N (T A)")
        else:
            state = state.permute(0, 2, 1)  # (N, A, T)
        return state
    
    
    def encode(self, state):
        # state: (B, T, A)
        state = self.preprocess(state / self.act_scale)
        state_rep = self.encoder(state)
        return state_rep
    
    
    def decode(self, state_rep):
        # state_rep: (B, D)
        if not self.use_mlp:
            state_rep = state_rep.unsqueeze(-1)  # (B, D, 1)
        dec_out = self.get_action_from_latent(state_rep) 
        return dec_out
        
        
    def forward(self, state):
        # state: (B, T, A)
        state = self.preprocess(state / self.act_scale)
        state_rep = self.encoder(state)
        dec_out = self.get_action_from_latent(state_rep)
        return dec_out
    
    
    def compute_loss(self, state):
        # state: (B, T, A)
        dec_out = self.forward(state)

        recon_loss = (state - dec_out).abs().mean()
        total_loss = recon_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.detach(),
            'vq_loss': torch.tensor(0.0)
        }
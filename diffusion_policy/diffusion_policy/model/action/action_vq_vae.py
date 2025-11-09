
"""
Adjust the code from https://github.com/jayLEE0301/vq_bet_official/blob/main/vector_quantize_pytorch/vector_quantize_pytorch.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit
import einops
import numpy as np

from diffusion_policy.model.common.residual_vq import ResidualVQ
from diffusion_policy.model.diffusion.conv1d_components import (
    Conv1dBlock, Downsample1d, Upsample1d
)
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        down_dims
    ):
        super(EncoderMLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, down_dims[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(down_dims)):
            layers.append(nn.Linear(down_dims[i-1], down_dims[i]))
            layers.append(nn.ReLU())

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
        kernel_size=3
    ):
        super(Encoder1DCNN, self).__init__()
        blocks = list()
        current_dim = input_dim
        for i, dim in enumerate(down_dims):
            blocks.append(Conv1dBlock(current_dim, dim, kernel_size=kernel_size))
            blocks.append(Downsample1d(dim))
            current_dim = dim
        self.encoder = nn.Sequential(*blocks)
        self.fc = nn.Linear(current_dim, output_dim)
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        return state

        
class DecoderConv1D(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 up_dims, 
                 kernel_size=3
        ):
        super().__init__()
        blocks = list()
        current_dim = input_dim
        for i, dim in enumerate(up_dims):
            blocks.append(Conv1dBlock(current_dim, dim, kernel_size=kernel_size))
            blocks.append(Upsample1d(dim))
            current_dim = dim
        self.decoder = nn.Sequential(*blocks)
        self.fc = nn.Conv1d(current_dim, output_dim, kernel_size=3, padding=1)
        self.apply(weights_init_encoder)

    def forward(self, x):
        # x: (B, input_dim, T_latent)
        h = self.decoder(x)
        x = self.fc(h)
        return x


class ActionVqVae(ModuleAttrMixin):
    def __init__(
        self,
        input_dim_h=16,  # length of action chunk
        input_dim_w=7,  # action dim
        down_dims=[128, 256],
        n_latent_dims=512,
        vqvae_n_embed=32,
        vqvae_groups=4,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
        use_mlp=True
    ):
        super(ActionVqVae, self).__init__()
        self.n_latent_dims = n_latent_dims
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.vqvae_n_embed = vqvae_n_embed
        self.vqvae_groups = vqvae_groups
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale
        self.use_mlp = use_mlp

        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}

        self.vq_layer = ResidualVQ(
            dim=self.n_latent_dims,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
        )
        self.embedding_dim = self.n_latent_dims

        if use_mlp:
            input_dim_w = input_dim_h * input_dim_w
            self.encoder = EncoderMLP(
                input_dim=input_dim_w, output_dim=n_latent_dims, 
                down_dims=down_dims
            )
            self.decoder = EncoderMLP(
                input_dim=n_latent_dims, output_dim=input_dim_w, 
                down_dims=list(reversed(down_dims))
            )
        else:
            self.encoder = Encoder1DCNN(
                input_dim=input_dim_w,
                output_dim=n_latent_dims,
                down_dims=down_dims
            )
            self.decoder = Encoder1DCNN(
                input_dim=n_latent_dims,
                output_dim=input_dim_w,
                down_dims=list(reversed(down_dims))
            )
            

    def get_action_from_latent(self, latent):
        with torch.no_grad():
            if self.use_mlp:
                output = self.decoder(latent) * self.act_scale
                # (B, T * A) -> (B, T, A)
                return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
            else:
                # latent: (B, D, T_latent)
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
        
        if self.use_mlp:
            # state_rep: (B, D)
            state_rep_flat = state_rep.unsqueeze(1) # (B, 1, D)
        else:
            # state_rep: (B, D, T_latent)
            state_rep_flat = state_rep.permute(0, 2, 1) # (B, T_latent, D)

        state_rep_flat, _, _ = self.vq_layer(state_rep_flat)
        
        if self.use_mlp:
            state_vq = state_rep_flat.squeeze(1) # (B, D)
        else:
            state_vq = state_rep_flat.permute(0, 2, 1) # (B, D, T_latent)

        return state_vq

        
    def forward(self, state):
        # state: (B, T, A)
        state = self.preprocess(state / self.act_scale)
        state_rep = self.encoder(state)
        
        if self.use_mlp:
            # state_rep: (B, D)
            state_rep_flat = state_rep.unsqueeze(1) # (B, 1, D)
        else:
            # state_rep: (B, D, T_latent)
            state_rep_flat = state_rep.permute(0, 2, 1) # (B, T_latent, D)

        state_rep_flat, vq_code, vq_loss = self.vq_layer(state_rep_flat)
        
        if self.use_mlp:
            state_vq = state_rep_flat.squeeze(1) # (B, D)
        else:
            state_vq = state_rep_flat.permute(0, 2, 1) # (B, D, T_latent)

        dec_out = self.get_action_from_latent(state_vq) * self.act_scale
        return dec_out, vq_loss
    
    
    def compute_loss(self, state):
        # state: (B, T, A)
        dec_out, vq_loss = self.forward(state)

        recon_loss = (state - dec_out).abs().mean()
        total_loss = recon_loss * self.encoder_loss_multiplier + vq_loss.sum() * 5.0
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss.detach(),
            'vq_loss': vq_loss.sum().detach()
        }
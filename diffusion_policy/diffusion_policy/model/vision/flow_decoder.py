import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from diffusion_policy.model.diffusion.conv2d_components import Upsample2d, Conv2dBlock

class FlowDecoder(nn.Module):
    """
    Decodes a sequence of latent vectors into a sequence of flow images.
    Input: (B, T, plan_dim)
    Output: (B, T, 2, H, W)
    """
    def __init__(self, 
            shape_meta: dict,
            plan_dim: int, 
            start_channels: int = 256,
            start_res: int = 6,  # The spatial resolution to start from
            upsample_channels: List[int] = [128, 64],
            kernel_size: int = 4,  # Use kernel_size=4 for 2x upsampling to avoid checkerboard
        ):
        super().__init__()
        flow_keys = list()
        for key, attr in shape_meta['flow'].items():
            flow_keys.append(key)
        assert len(flow_keys) == 1, "Currently only supports one flow output"
        flow_shape = shape_meta['flow'][flow_keys[0]]['shape']  # e.g., (2, 84, 84)
        out_channels = flow_shape[0] 
        
        # 1. A linear layer to project the plan vector into a starting 2D feature map
        self.proj = nn.Linear(plan_dim, start_channels * start_res * start_res)
        
        # 2. The main upsampling network
        layers = list()
        in_channels = start_channels
        
        # Add upsampling layers
        for i, c in enumerate(upsample_channels):
            layers.append(Upsample2d(in_channels))  # Upsample2d uses ConvTranspose2d
            layers.append(Conv2dBlock(in_channels, c, kernel_size=kernel_size))
            in_channels = c
            
        # Final layer to get to the desired output channels (e.g., 2 for flow)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        
        self.net = nn.Sequential(*layers)
        
        # Store attributes for reshaping
        self.start_channels = start_channels
        self.start_res = start_res
        self.output_shape = flow_shape[1:]
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, T, plan_dim)
        """
        B, T, _ = x.shape
        
        # (B, T, plan_dim) -> (B * T, plan_dim)
        x = x.reshape(B * T, -1)
        
        # Project to a starting feature map
        # (B * T, plan_dim) -> (B * T, start_channels * start_res * start_res)
        x = self.proj(x)
        
        # Reshape to a 2D feature map
        # (B * T, ...) -> (B * T, start_channels, start_res, start_res)
        x = x.reshape(B * T, self.start_channels, self.start_res, self.start_res)
        
        # Pass through the upsampling network
        # (B * T, C_in, H_in, W_in) -> (B * T, C_out, H_out, W_out)
        x = self.net(x)
        
        # Interpolate to the final output size on H, W dimensions
        x = F.interpolate(x, size=self.output_shape, mode='bilinear', align_corners=False)

        # Reshape back to include the time dimension
        # (B * T, C_out, H_out, W_out) -> (B, T, C_out, H_out, W_out)
        _, C_out, H_out, W_out = x.shape
        x = x.reshape(B, T, C_out, H_out, W_out)
        
        return x
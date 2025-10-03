import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Tuple
from diffusion_policy.model.common.flovd_module import TemporalTransformerBlock
from diffusion_policy.common.flow_util import InputPadder



def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk is False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk is False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down:
            self.down_opt = Downsample(in_c, use_conv=use_conv)
            self.down_skip = Downsample(in_c, use_conv=use_conv)
        else:
            self.down_opt = None
            self.down_skip = None

    def forward(self, x):
        residual = x
        if self.down:
            x = self.down_opt(x)
            residual = self.down_skip(residual)
        if self.in_conv is not None:
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            residual = self.skep(residual)
        return h + residual

# class ResnetBlock(nn.Module):

#     def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
#         super().__init__()
#         ps = ksize // 2
#         if in_c != out_c or sk == False:
#             self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
#         else:
#             self.in_conv = None
#         self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
#         self.act = nn.ReLU()
#         self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
#         if sk == False:
#             self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
#         else:
#             self.skep = None

#         self.down = down
#         if self.down == True:
#             self.down_opt = Downsample(in_c, use_conv=use_conv)

#     def forward(self, x):
#         if self.down == True:
#             x = self.down_opt(x)
#         if self.in_conv is not None:  # edit
#             x = self.in_conv(x)

#         h = self.block1(x)
#         h = self.act(h)
#         h = self.block2(h)
#         if self.skep is not None:
#             return h + self.skep(x)
#         else:
#             return h + x


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            d_model,
            dropout=0.,
            max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2, ...] = torch.sin(position * div_term)
        pe[0, :, 1::2, ...] = torch.cos(position * div_term)
        pe.unsqueeze_(-1).unsqueeze_(-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), ...]
        return self.dropout(x)

# Code based on CameraPoseEncoder from CameraCtrl (https://github.com/hehao13/CameraCtrl/tree/svd)
# Modified by Wonjoon Jin.
class FlowEncoder(nn.Module):
    def __init__(self,
                 downscale_factor,
                 channels,
                 input_dims=(84, 84),
                 nums_rb=3,
                 ksize=3,
                 sk=False,
                 use_conv=True,
                 compression_factor=1,
                 temporal_attention_nhead=8,
                 attention_block_types=("Temporal_Self", ),
                 temporal_position_encoding=False,
                 temporal_position_encoding_max_len=16,
                 rescale_output_factor=1.0):
        super(FlowEncoder, self).__init__()
        self.padder = InputPadder(input_dims, factor=downscale_factor)
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        # compute input channel after unshuffle
        cin = 3 * (downscale_factor ** 2)
        self.encoder_conv_in = conv_nd(2, cin, channels[0], ksize, 1, ksize // 2)
        
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        self.encoder_down_conv_out_blocks = nn.ModuleList()
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList()
            in_c = channels[i - 1] if i != 0 else channels[0]
            for j in range(nums_rb):
                if j < nums_rb - 1:
                    out_c = channels[i] // compression_factor
                else:
                    out_c = channels[i]
                is_downsampling = (j == 0 and i != 0)
                conv_layer = ResnetBlock(in_c, out_c, down=is_downsampling, ksize=ksize, sk=sk, use_conv=use_conv)
                # temporal_attention_layer = TemporalTransformerBlock(dim=out_c,
                #                                                     num_attention_heads=temporal_attention_nhead,
                #                                                     attention_head_dim=(out_c // temporal_attention_nhead),
                #                                                     attention_block_types=attention_block_types,
                #                                                     dropout=0.0,
                #                                                     cross_attention_dim=None,
                #                                                     temporal_position_encoding=temporal_position_encoding,
                #                                                     temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                #                                                     rescale_output_factor=rescale_output_factor)
                conv_layers.append(conv_layer)
                # temporal_attention_layers.append(temporal_attention_layer)
                
                # update in_c to the output of the current block
                in_c = out_c
            
            # ZeroConv out block
            conv_out_block = nn.Conv2d(in_c, in_c, 1, 1, 0)
            torch.nn.init.zeros_(conv_out_block.weight)
            torch.nn.init.zeros_(conv_out_block.bias)
            
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(temporal_attention_layers)
            self.encoder_down_conv_out_blocks.append(conv_out_block)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        # unshuffle
        bs = x.shape[0]
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.padder.pad(x)
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.encoder_conv_in(x)
        for res_block, attention_block, conv_out_block in zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks, self.encoder_down_conv_out_blocks):
            # for res_layer, attention_layer in zip(res_block, attention_block):
            for res_layer in res_block:
                x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, '(b f) c h w -> (b h w) f c', b=bs)
                # x = attention_layer(x)
                x = rearrange(x, '(b h w) f c -> (b f) c h w', h=h, w=w)
            
            # ZeroConv out block
            feature = conv_out_block(x)
            features.append(rearrange(feature, '(b f) c h w -> b c f h w', b=bs))
        return features
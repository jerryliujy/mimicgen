import torch
import torch.nn as nn
import math
from typing import Dict, Callable, Optional, Any
from einops import rearrange
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention


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
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



class TemporalSelfAttention(Attention):
    def __init__(
            self,
            attention_mode=None,
            temporal_position_encoding=False,
            temporal_position_encoding_max_len=32,
            rescale_output_factor=1.0,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal_Self"

        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None
        self.rescale_output_factor = rescale_output_factor

    def set_use_memory_efficient_attention_xformers(
            self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        # disable motion module efficient xformers to avoid bad results, don't know why
        # TODO: fix this bug
        pass

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        # add position encoding
        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)
        if "pose_feature" in cross_attention_kwargs:
            pose_feature = cross_attention_kwargs["pose_feature"]
            if pose_feature.ndim == 5:
                pose_feature = rearrange(pose_feature, "b c f h w -> (b h w) f c")
            else:
                assert pose_feature.ndim == 3
            cross_attention_kwargs["pose_feature"] = pose_feature

        if isinstance(self.processor,  PoseAdaptorAttnProcessor):
            return self.processor(
                self,
                hidden_states,
                cross_attention_kwargs.pop('pose_feature'),
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        elif hasattr(self.processor, "__call__"):
            return self.processor.__call__(
                    self,
                    hidden_states,
                    encoder_hidden_states=None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
        else:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            

class TemporalTransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_attention_heads,
            attention_head_dim,
            attention_block_types=("Temporal_Self", "Temporal_Self",),
            dropout=0.0,
            norm_num_groups=32,
            cross_attention_dim=768,
            activation_fn="geglu",
            attention_bias=False,
            upcast_attention=False,
            temporal_position_encoding=False,
            temporal_position_encoding_max_len=32,
            encoder_hidden_states_query=(False, False),
            attention_activation_scale=1.0,
            attention_processor_kwargs: Dict = {},
            rescale_output_factor=1.0
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        self.attention_block_types = attention_block_types

        for block_idx, block_name in enumerate(attention_block_types):
            attention_blocks.append(
                TemporalSelfAttention(
                    attention_mode=block_name,
                    cross_attention_dim=cross_attention_dim if block_name in ['Temporal_Cross', 'Temporal_Pose_Adaptor'] else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rescale_output_factor=rescale_output_factor,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs: Dict[str, Any] = {}):
        for attention_block, norm, attention_block_type in zip(self.attention_blocks, self.norms, self.attention_block_types):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=norm_hidden_states if attention_block_type == 'Temporal_Self' else encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs
            ) + hidden_states

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output



class PoseAdaptorAttnProcessor(nn.Module):
    def __init__(self,
                 hidden_size,  # dimension of hidden state
                 pose_feature_dim=None,  # dimension of the pose feature
                 cross_attention_dim=None,  # dimension of the text embedding
                 query_condition=False,
                 key_value_condition=False,
                 scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.pose_feature_dim = pose_feature_dim
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.query_condition = query_condition
        self.key_value_condition = key_value_condition
        assert hidden_size == pose_feature_dim
        
        # Do Not Use merge layer from CameraCtrl.
        if self.query_condition and self.key_value_condition:
            # self.qkv_merge = nn.Linear(hidden_size, hidden_size)
            # init.zeros_(self.qkv_merge.weight)
            # init.zeros_(self.qkv_merge.bias)
            self.qkv_merge = nn.Identity()
        elif self.query_condition:
            # self.q_merge = nn.Linear(hidden_size, hidden_size)
            # init.zeros_(self.q_merge.weight)
            # init.zeros_(self.q_merge.bias)
            self.q_merge = nn.Identity()
        else:
            # self.kv_merge = nn.Linear(hidden_size, hidden_size)
            # init.zeros_(self.kv_merge.weight)
            # init.zeros_(self.kv_merge.bias)
            self.kv_merge = nn.Identity()

    def forward(self,
                attn,
                hidden_states,
                pose_feature,
                encoder_hidden_states=None,
                attention_mask=None,
                temb=None,
                scale=None,):
        assert pose_feature is not None
        pose_embedding_scale = (scale or self.scale)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        assert hidden_states.ndim == 3 and pose_feature.ndim == 3

        if self.query_condition and self.key_value_condition:
            assert encoder_hidden_states is None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        assert encoder_hidden_states.ndim == 3

        batch_size, ehs_sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, ehs_sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # import pdb
        # pdb.set_trace()
        # if self.query_condition and self.key_value_condition:  # only self attention
        #     query_hidden_state = self.qkv_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
        #     key_value_hidden_state = query_hidden_state
        # elif self.query_condition:
        #     query_hidden_state = self.q_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
        #     key_value_hidden_state = encoder_hidden_states
        # else:
        #     key_value_hidden_state = self.kv_merge(encoder_hidden_states + pose_feature) * pose_embedding_scale + encoder_hidden_states
        #     query_hidden_state = hidden_states
        
        # Merge pose_features. just by adding.
        if self.query_condition and self.key_value_condition:  # only self attention
            query_hidden_state = hidden_states + pose_feature * pose_embedding_scale
            key_value_hidden_state = query_hidden_state
        elif self.query_condition:
            query_hidden_state = hidden_states + pose_feature * pose_embedding_scale
            key_value_hidden_state = encoder_hidden_states
        else:
            key_value_hidden_state = encoder_hidden_states + pose_feature * pose_embedding_scale
            query_hidden_state = hidden_states

        # original attention
        query = attn.to_q(query_hidden_state)
        key = attn.to_k(key_value_hidden_state)
        value = attn.to_v(key_value_hidden_state)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
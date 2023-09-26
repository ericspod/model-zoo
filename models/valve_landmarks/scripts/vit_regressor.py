from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import ViT
from monai.utils import ensure_tuple, ensure_tuple_rep

from monai.networks.nets import Regressor
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks import ConvDenseBlock, Convolution
from monai.networks.layers import Flatten, Reshape
from monai.networks.layers.factories import Act, Norm

class ParallelCat(nn.Module):
    """
    Apply the same input to each of the given modules and concatenate their results together.

    Args:
        catmodules: sequence of nn.Module objects to apply inputs to
        cat_dim: dimension to concatenate along when combining outputs
    """

    def __init__(self, catmodules: Sequence[nn.Module], cat_dim: int = 1, add_channel=False):
        super().__init__()
        self.cat_dim = cat_dim
        self.add_channel=add_channel

        for i, s in enumerate(catmodules):
            self.add_module(f"catmodule_{i}", s)

    def forward(self, x):
        tensors = [s(x) for s in self.children()]
        
        if self.add_channel:
            tensors = [t[..., None] for t in tensors]
            
        return torch.cat(tensors, self.cat_dim)

class ViTRegressorV4(Regressor):
    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: float | None = None,
        bias: bool = True,
        patch_size: Sequence[int] | int = 16,
        hidden_size: int = 256,
        mlp_dim: int = 1024,
        num_trans_layers: int = 1,
        num_heads: int = 16,
        pos_embed: str = "conv",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        self.patch_size = ensure_tuple_rep(patch_size, len(in_shape) - 1)
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_trans_layers = num_trans_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.qkv_bias = qkv_bias
        self.save_attn = save_attn

        super().__init__(in_shape, out_shape, channels, strides, kernel_size, num_res_units, act, norm, dropout, bias)

    def _get_layer(self, in_channels, out_channels, strides, is_last):
        dout = out_channels - in_channels
        dilations = [1, 2, 4]
        dchannels = [dout // 3, dout // 3, dout // 3 + dout % 3]

        db = ConvDenseBlock(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            channels=dchannels,
            dilations=dilations,
            kernel_size=self.kernel_size,
            num_res_units=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
        )

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=out_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )

        return nn.Sequential(db, conv)

    def _get_final_layer(self, in_shape):
        dropout = 0 if self.dropout is None else self.dropout
        tout_size = int(self.hidden_size * np.prod([i // p for i, p in zip(in_shape[1:], self.patch_size)]))
        targs = (self.hidden_size, self.mlp_dim, self.num_heads, dropout, self.qkv_bias, self.save_attn)
        
        point_paths = []
        
        for _ in range(self.out_shape[1]):
            conv = Convolution(
                spatial_dims=self.dimensions,
                in_channels=in_shape[0],
                out_channels=in_shape[0],
                strides=1,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                conv_only=True,
            )
            
            patch_embedding = PatchEmbeddingBlock(
                in_channels=in_shape[0],
                img_size=in_shape[1:],
                patch_size=self.patch_size,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                pos_embed=self.pos_embed,
                dropout_rate=dropout,
                spatial_dims=len(in_shape) - 1,
            )
            
            blocks = [TransformerBlock(*targs) for i in range(self.num_trans_layers)]
            norm = nn.LayerNorm(self.hidden_size)
            linear = nn.Linear(tout_size, self.out_shape[0])
            point_paths.append(nn.Sequential(conv,patch_embedding,*blocks,norm,Flatten(),linear))

        return torch.nn.Sequential(ParallelCat(point_paths,-1,True))
    

class ViTRegressorV3(Regressor):
    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: float | None = None,
        bias: bool = True,
        patch_size: Sequence[int] | int = 16,
        hidden_size: int = 256,
        mlp_dim: int = 1024,
        num_trans_layers: int = 4,
        num_heads: int = 16,
        pos_embed: str = "conv",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        self.patch_size = ensure_tuple_rep(patch_size, len(in_shape) - 1)
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_trans_layers = num_trans_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.qkv_bias = qkv_bias
        self.save_attn = save_attn

        super().__init__(in_shape, out_shape, channels, strides, kernel_size, num_res_units, act, norm, dropout, bias)

    def _get_layer(self, in_channels, out_channels, strides, is_last):
        dout = out_channels - in_channels
        dilations = [1, 2, 4]
        dchannels = [dout // 3, dout // 3, dout // 3 + dout % 3]

        db = ConvDenseBlock(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            channels=dchannels,
            dilations=dilations,
            kernel_size=self.kernel_size,
            num_res_units=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
        )

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=out_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )

        return nn.Sequential(db, conv)

    def _get_final_layer(self, in_shape):
        dropout = 0 if self.dropout is None else self.dropout
        patch_embedding = PatchEmbeddingBlock(
            in_channels=in_shape[0],
            img_size=in_shape[1:],
            patch_size=self.patch_size,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            pos_embed=self.pos_embed,
            dropout_rate=dropout,
            spatial_dims=len(in_shape) - 1,
        )

        targs = (self.hidden_size, self.mlp_dim, self.num_heads, dropout, self.qkv_bias, self.save_attn)
        blocks = [TransformerBlock(*targs) for i in range(self.num_trans_layers)]

        norm = nn.LayerNorm(self.hidden_size)

        out_size = int(self.hidden_size * np.prod([i // p for i, p in zip(in_shape[1:], self.patch_size)]))

        linear = nn.Linear(out_size, np.prod(self.out_shape))

        return nn.Sequential(patch_embedding, *blocks, norm, Flatten(), linear, Reshape(*self.out_shape))


# class ParallelCat(nn.Module):
#     """
#     Apply the same input to each of the given modules and concatenate their results together.

#     Args:
#         catmodules: sequence of nn.Module objects to apply inputs to
#         cat_dim: dimension to concatenate along when combining outputs
#     """

#     def __init__(self, catmodules: Sequence[nn.Module], cat_dim: int = 1):
#         super().__init__()
#         self.cat_dim = cat_dim

#         for i, s in enumerate(catmodules):
#             self.add_module(f"catmodule_{i}", s)

#     def forward(self, x):
#         tensors = [s(x) for s in self.children()]
#         return torch.cat(tensors, self.cat_dim)


# class ViTRegressorV2(nn.Module):
#     """
#     Vision Transformer (ViT), based on: "Dosovitskiy et al.,
#     An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

#     ViT supports Torchscript but only works for Pytorch after 1.8.
#     """

#     def __init__(
#         self,
#         in_shape: Sequence[int],
#         out_shape: Sequence[int],
#         patch_size: Sequence[int] | int,
#         hidden_size: int = 768,
#         mlp_dim: int = 3072,
#         num_layers: int = 12,
#         num_heads: int = 12,
#         pos_embed: str = "conv",
#         dropout_rate: float = 0.0,
#         qkv_bias: bool = False,
#         save_attn: bool = False,
#     ) -> None:

#         super().__init__()
#         self.out_shape = ensure_tuple(out_shape)
#         num_classes = np.product(self.out_shape)

#         in_channels, *img_size=in_shape

#         self.patch_embedding = PatchEmbeddingBlock(
#             in_channels=in_channels,
#             img_size=img_size,
#             patch_size=patch_size,
#             hidden_size=hidden_size,
#             num_heads=num_heads,
#             pos_embed=pos_embed,
#             dropout_rate=dropout_rate,
#             spatial_dims=len(img_size),
#         )
#         self.blocks = nn.Sequential(
#             *[TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
#             for i in range(num_layers)]
#         )
#         # self.final_layer = nn.Linear(hidden_size, num_classes)  # type: ignore

#         point_paths = [nn.Linear(hidden_size, self.out_shape[0]) for _ in range(self.out_shape[1])]
#         self.final_layer = ParallelCat(point_paths)

#         for p in self.parameters():
#             if p.ndim==1:
#                 torch.nn.init.normal_(p)
#             else:
#                 torch.nn.init.kaiming_normal_(p)

#     def forward(self, x):
#         x = self.patch_embedding(x)
#         x = self.blocks(x)
#         x = self.final_layer(x[:, 0])
#         return x.reshape((x.shape[0],) + self.out_shape)

# class ViTRegressorV1(ViT):
#     def __init__(
#         self,
#         # in_channels: int,
#         # img_size: Sequence[int] | int,
#         in_shape: Sequence[int],
#         out_shape: Sequence[int],
#         patch_size: Sequence[int] | int,
#         hidden_size: int = 768,
#         mlp_dim: int = 3072,
#         num_layers: int = 12,
#         num_heads: int = 12,
#         pos_embed: str = "conv",
#         # classification: bool = False,
#         # num_classes: int = 2,
#         dropout_rate: float = 0.0,
#         # spatial_dims: int = 3,
#         # post_activation="Tanh",
#         qkv_bias: bool = False,
#         save_attn: bool = False,
#     ) -> None:
#         out_shape = ensure_tuple(out_shape)
#         num_classes = np.product(out_shape)
#         super().__init__(
#             in_shape[0],
#             in_shape[1:],
#             patch_size,
#             hidden_size,
#             mlp_dim,
#             num_layers,
#             num_heads,
#             pos_embed,
#             True,
#             num_classes,
#             dropout_rate,
#             len(in_shape) - 1,
#             None,
#             qkv_bias,
#             save_attn,
#         )
#         self.out_shape = out_shape
#         self.norm = nn.Identity()

#     def forward(self, x):
#         out, _ = super().forward(x)
#         return out.reshape((x.shape[0],) + self.out_shape)

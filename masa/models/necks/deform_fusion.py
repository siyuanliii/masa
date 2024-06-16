"""
Author: Siyuan Li
Licensed: Apache-2.0 License
"""

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmdet.registry import MODELS
from mmengine.model import BaseModule, constant_init, normal_init

# Reference:
# https://github.com/microsoft/DynamicHead
# https://github.com/jshilong/SEPC


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return rearrange(x, "b h w c -> b c h w")


class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.

    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        norm_cfg=dict(type="GN", num_groups=16, requires_grad=True),
    ):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias
        )
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x


class DyHeadBlock(nn.Module):
    """Modified DyHead Block for dynamic feature fusion.
       We remove the task and scale aware attention in the original implementation.

    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
    """

    def __init__(
        self, in_channels, out_channels, zero_init_offset=True, fix_upsample=False,
    ):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        self.fix_upsample = fix_upsample
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3 * 3
        self.offset_dim = 3 * 2 * 3 * 3

        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1
        )
        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, : self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid()

            # calculate offset and mask of DCNv2 from current feature
            offsets = offset.split(offset.size(1) // 3, dim=1)
            masks = mask.split(mask.size(1) // 3, dim=1)

            sum_feat = self.spatial_conv_mid(x[level], offsets[0], masks[0])
            summed_levels = 1
            if level > 0:
                sum_feat += self.spatial_conv_low(x[level - 1], offsets[1], masks[1])
                summed_levels += 1
            if level < len(x) - 1:
                if not self.fix_upsample:
                    # this upsample order is weird, but faster than natural order
                    # https://github.com/microsoft/DynamicHead/issues/25
                    sum_feat += F.interpolate(
                        self.spatial_conv_high(x[level + 1], offsets[2], masks[2]),
                        size=x[level].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    sum_feat += self.spatial_conv_high(
                        F.interpolate(
                            x[level + 1],
                            size=x[level].shape[-2:],
                            mode="bilinear",
                            align_corners=True,
                        ),
                        offsets[2],
                        masks[2],
                    )
                summed_levels += 1
            outs.append(sum_feat / summed_levels)

        return outs

@MODELS.register_module()
class DeformFusion(BaseModule):
    """Deformable Fusion Module for MASA."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=6,
        zero_init_offset=True,
        fix_upsample=False,
        init_cfg=None,
    ):
        assert init_cfg is None, (
            "To prevent abnormal initialization "
            "behavior, init_cfg is not allowed to be set"
        )
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.zero_init_offset = zero_init_offset

        dyhead_blocks = []
        for i in range(num_blocks):
            in_channels = self.in_channels if i == 0 else self.out_channels
            dyhead_blocks.append(
                DyHeadBlock(
                    in_channels,
                    self.out_channels,
                    zero_init_offset=zero_init_offset,
                    fix_upsample=fix_upsample,
                )
            )
        self.dyhead_blocks = nn.Sequential(*dyhead_blocks)

    def forward(self, inputs):
        """Forward function."""
        assert isinstance(inputs, (tuple, list))
        outs = self.dyhead_blocks(inputs)
        return tuple(outs)

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from fcos_core.utils.flops import count_apply_module, count_no_apply_module

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.lateral_blocks = []
        self.fpn_blocks = []
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_lateral{}".format(idx)
            layer_block = "fpn_out{}".format(idx)

            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1)
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.lateral_blocks.append(inner_block)
            self.fpn_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward_dummy(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        self.__flops__ = 0
        last_inner, flops = count_apply_module(getattr(self, self.lateral_blocks[-1]), [x[-1]])
        self.__flops__ += flops

        results = []
        tmp, flops = count_apply_module(getattr(self, self.fpn_blocks[-1]), [last_inner])
        results.append(tmp)
        self.__flops__ += flops

        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.lateral_blocks[:-1][::-1], self.fpn_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral, flops = count_apply_module(getattr(self, inner_block), [feature])
            self.__flops__ += flops
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            self.__flops__ += last_inner.numel()
            tmp, flops = count_apply_module(getattr(self, layer_block), [last_inner])
            self.__flops__ += flops
            results.insert(0, tmp)

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks.forward_dummy(x[-1], results[-1])
            results.extend(last_results)
            self.__flops__ += self.top_blocks.__flops__
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks.forward_dummy(results[-1])
            results.extend(last_results)
            self.__flops__ += self.top_blocks.__flops__

        return tuple(results)

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        last_inner = getattr(self, self.lateral_blocks[-1])(x[-1])
        results = []
        results.append(getattr(self, self.fpn_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.lateral_blocks[:-1][::-1], self.fpn_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")
            inner_lateral = getattr(self, inner_block)(feature)
            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)
            last_inner = inner_lateral + inner_top_down
            results.insert(0, getattr(self, layer_block)(last_inner))

        if isinstance(self.top_blocks, LastLevelP6P7):
            last_results = self.top_blocks(x[-1], results[-1])
            results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]

    def forward_dummy(self, x):
        tmp = F.max_pool2d(x, 1, 2, 0)
        self.__flops__ = count_no_apply_module(
            nn.MaxPool2d(kernel_size=1, stride=2, padding=0),
            [x], [tmp]
        )
        return [tmp]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels):
        super(LastLevelP6P7, self).__init__()
        self.conv_p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.conv_p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.conv_p6, self.conv_p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.conv_p6(x)
        p7 = self.conv_p7(F.relu(p6))
        return [p6, p7]

    def forward_dummy(self, c5, p5):
        self.__flops__ = 0
        x = p5 if self.use_P5 else c5

        p6, flops = count_apply_module(self.conv_p6, [x])
        self.__flops__ += flops

        p6_ = F.relu(p6)
        self.__flops__ += p6.numel()

        p7, flops = count_apply_module(self.conv_p7, [p6_])
        self.__flops__ += flops

        return [p6, p7]
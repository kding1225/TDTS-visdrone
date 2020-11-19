# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
import torch.nn as nn
from fcos_core.layers import FrozenBatchNorm2d


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    assert isinstance(conv, torch.nn.Conv2d)
    assert isinstance(bn, (torch.nn.BatchNorm2d, FrozenBatchNorm2d))
    if isinstance(bn, FrozenBatchNorm2d):
        bn.eps = 0.

    device = conv.weight.device
    bn = bn.to(device)

    print('before fuse:')
    print(conv, bn)

    with torch.no_grad():
        # init
        fusedconv = torch.nn.Conv2d(conv.in_channels,
                                    conv.out_channels,
                                    kernel_size=conv.kernel_size,
                                    stride=conv.stride,
                                    padding=conv.padding,
                                    bias=True)
        fusedconv.to(device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0)).to(device)
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
            torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        print('after fuse:')
        print(fusedconv)
        print()

        return fusedconv
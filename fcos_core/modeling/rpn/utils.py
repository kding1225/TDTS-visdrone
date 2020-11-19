# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Utility functions minipulating the prediction layers
"""

from ..utils import cat

import torch
import torch.nn as nn


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight'):
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight'):
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def meshgrid(h, w, stride, device, dtype=torch.float32):
    """
    generate grid coordinates
    """
    shifts_x = torch.arange(
        0, w, step=1,
        dtype=dtype, device=device
    )*stride
    shifts_y = torch.arange(
        0, h, step=1,
        dtype=dtype, device=device
    )*stride
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    return shift_y, shift_x


def make_checkboard(h, w, device):
    y, x = meshgrid(h, w, 1, device, dtype=torch.int)
    S = torch.empty(h, w, device=device, dtype=torch.long)
    # import ipdb; ipdb.set_trace()

    # 0
    mask = ((x % 2 == 0) & (y % 2 == 0)) | (((x - 1) % 2 == 0) & ((y - 1) % 2 == 0))
    S[mask] = 0

    # 1
    mask = ((y % 4 == 0) & ((x - 1) % 4 == 0)) | (((y - 2) % 4 == 0) & ((x - 3) % 4 == 0))
    S[mask] = 1

    # 2
    mask = ((y % 4 == 0) & ((x - 3) % 4 == 0)) | (((y - 2) % 4 == 0) & ((x - 1) % 4 == 0))
    S[mask] = 2

    # 3
    mask = (((y - 1) % 4 == 0) & (x % 4 == 0)) | (((y - 3) % 4 == 0) & ((x - 2) % 4 == 0))
    S[mask] = 3

    # 4
    mask = (((y - 1) % 4 == 0) & ((x - 2) % 4 == 0)) | (((y - 3) % 4 == 0) & (x % 4 == 0))
    S[mask] = 4

    return S


def smooth_min_max(x, y=None):
    """
    x: tensor, (n,2) when y is None
    y: None or tensor (n,)
    """

    if y is None:
        z = x
    else:
        z = torch.cat([x[:,None], y[:,None]], dim=1)

    s = torch.exp(z-z.max(dim=1, keepdim=True)[0])
    s = s/s.sum(dim=1, keepdim=True)

    z_max = (z*s).sum(dim=1)
    z_min = (z[:,[1,0]]*s).sum(dim=1)

    return z_min, z_max
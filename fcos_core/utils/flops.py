import numpy as np
import torch.nn as nn

from fcos_core.modeling.rpn.tdts_visdrone.utils import ChannelNorm as ChannelNorm1
from fcos_core.modeling.rpn.fcos_visdrone.utils import ChannelNorm as ChannelNorm2
from fcos_core.modeling.rpn.fcos_visdrone.utils import NoopLayer as NoopLayer1
from fcos_core.modeling.rpn.tdts_visdrone.utils import NoopLayer as NoopLayer2
from spconv import SubMConv2d
from fcos_core.layers.batch_norm import FrozenBatchNorm2d


def flops_to_string(flops, units='GFLOPs', precision=2):
    """Convert FLOPs number into a string.
    Note that Here we take a multiply-add counts as one FLOP.
    Args:
        flops (float): FLOPs number to be converted.
        units (str | None): Converted FLOPs units. Options are None, 'GFLOPs',
            'MFLOPs', 'KFLOPs', 'FLOPs'. If set to None, it will automatically
            choose the most suitable unit for FLOPs. Default: 'GFLOPs'.
        precision (int): Digit number after the decimal point. Default: 2.
    Returns:
        str: The converted FLOPs number with units.
    Examples:
        >>> flops_to_string(1e9)
        '1.0 GFLOPs'
        >>> flops_to_string(2e5, 'MFLOPs')
        '0.2 MFLOPs'
        >>> flops_to_string(3e-9, None)
        '3e-09 FLOPs'
    """
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GFLOPs'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MFLOPs'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KFLOPs'
        else:
            return str(flops) + ' FLOPs'
    else:
        if units == 'GFLOPs':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'BFLOPs':
            return str(round(flops / 10. ** 9, precision)) + ' ' + units
        elif units == 'MFLOPs':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KFLOPs':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' FLOPs'


def params_to_string(num_params, units=None, precision=2):
    """Convert parameter number into a string.
    Args:
        num_params (float): Parameter number to be converted.
        units (str | None): Converted FLOPs units. Options are None, 'M',
            'K' and ''. If set to None, it will automatically choose the most
            suitable unit for Parameter number. Default: None.
        precision (int): Digit number after the decimal point. Default: 2.
    Returns:
        str: The converted parameter number with units.
    Examples:
        >>> params_to_string(1e9)
        '1000.0 M'
        >>> params_to_string(2e5)
        '200.0 k'
        >>> params_to_string(3e-9)
        '3e-09'
    """
    if units is None:
        if num_params // 10**6 > 0:
            return str(round(num_params / 10**6, precision)) + ' M'
        elif num_params // 10**3:
            return str(round(num_params / 10**3, precision)) + ' k'
        else:
            return str(num_params)
    else:
        if units == 'M':
            return str(round(num_params / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(num_params / 10.**3, precision)) + ' ' + units
        else:
            return str(num_params)


def gn_flops_counter(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    return int(batch_flops)


def channelnorm_flops_counter(module, input, output):
    cn_flops = int(2*np.prod(input[0].shape))
    return int(cn_flops)


def submconv2d_flops_counter(module, input, output):
    input = input[0].features

    kernel_dims = list(module.kernel_size)
    in_channels = module.in_channels
    out_channels = module.out_channels
    groups = module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = input.size(0)
    overall_conv_flops = conv_per_position_flops * active_elements_count  # todo: slightly larger than actual

    bias_flops = 0
    if module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    return int(overall_flops)


def upsample_flops_counter(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    return int(output_elements_count)


def relu_flops_counter(module, input, output):
    active_elements_count = input[0].numel()
    return int(active_elements_count)


def linear_flops_counter(module, input, output):
    input = input[0]
    output = output[0]
    output_last_dim = output.shape[
        -1]  # pytorch checks dimensions, so here we don't care much
    return int(np.prod(input.shape) * output_last_dim)


def pool_flops_counter(module, input, output):
    input = input[0]
    return int(np.prod(input.shape))


def bn_flops_counter(module, input, output):
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    return int(batch_flops)


def fbn_flops_counter(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    return int(batch_flops)


def noop_flops_counter(module, input, output):
    return 0


def conv_flops_counter(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    output = output[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


def count_apply_module(module, input):
    output = module(*input)
    if not isinstance(output, (list, tuple)):
        output = [output]
    flag = False
    for m in FLOPS_TAB:
        if isinstance(module, m):
            flops = FLOPS_TAB[m](module, input, output)
            flag = True
            break
    if not flag:
        raise NotImplementedError(module)
    return (*output, flops)


def count_no_apply_module(module, input, output):
    flag = False
    for m in FLOPS_TAB:
        if isinstance(module, m):
            flops = FLOPS_TAB[m](module, input, output)
            flag = True
            break
    if not flag:
        raise NotImplementedError(module)
    return flops


FLOPS_TAB = {
    # convolutions
    nn.Conv1d: conv_flops_counter,
    nn.Conv2d: conv_flops_counter,
    nn.Conv3d: conv_flops_counter,
    # activations
    nn.ReLU: relu_flops_counter,
    nn.PReLU: relu_flops_counter,
    nn.ELU: relu_flops_counter,
    nn.LeakyReLU: relu_flops_counter,
    nn.ReLU6: relu_flops_counter,
    # poolings
    nn.MaxPool1d: pool_flops_counter,
    nn.AvgPool1d: pool_flops_counter,
    nn.AvgPool2d: pool_flops_counter,
    nn.MaxPool2d: pool_flops_counter,
    nn.MaxPool3d: pool_flops_counter,
    nn.AvgPool3d: pool_flops_counter,
    nn.AdaptiveMaxPool1d: pool_flops_counter,
    nn.AdaptiveAvgPool1d: pool_flops_counter,
    nn.AdaptiveMaxPool2d: pool_flops_counter,
    nn.AdaptiveAvgPool2d: pool_flops_counter,
    nn.AdaptiveMaxPool3d: pool_flops_counter,
    nn.AdaptiveAvgPool3d: pool_flops_counter,
    # BNs
    nn.BatchNorm1d: bn_flops_counter,
    nn.BatchNorm2d: bn_flops_counter,
    nn.BatchNorm3d: bn_flops_counter,
    FrozenBatchNorm2d: fbn_flops_counter,
    # FC
    nn.Linear: linear_flops_counter,
    # Upscale
    nn.Upsample: upsample_flops_counter,
    # ChannelNorm
    ChannelNorm1: channelnorm_flops_counter,
    ChannelNorm2: channelnorm_flops_counter,
    # SubMConv2d
    SubMConv2d: submconv2d_flops_counter,
    # GroupNorm
    nn.GroupNorm: gn_flops_counter,
    # NoopLayer
    NoopLayer1: noop_flops_counter,
    NoopLayer2: noop_flops_counter
}
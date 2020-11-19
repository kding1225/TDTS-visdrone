
""" MobileNet V3

A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.

Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244

Hacked together by Ross Wightman
"""

from .efficientnets.efficientnet_builder import *
from .efficientnets.activations import HardSwish, hard_sigmoid
from .efficientnets.conv2d_layers import select_conv2d
from .efficientnets.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, \
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from fcos_core.utils.registry import Registry
from fcos_core.layers import FrozenBatchNorm2d

MNV3 = Registry()
_DEBUG = False


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


# need to set xml according to the flowing info
default_cfgs = {
    'tf_mobilenetv3_large_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_100': _cfg(
        url= 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
}


class MobileNetV3(nn.Module):
    """
    MobileNetV3 Feature Extractor
    """

    def __init__(self, cfg):
        super(MobileNetV3, self).__init__()

        if cfg.MODEL.MNV3.NORM_FUNC == 'FrozenBatchNorm2d':
            norm_func = FrozenBatchNorm2d
        elif cfg.MODEL.MNV3.NORM_FUNC == 'BatchNorm2d':
            norm_func = nn.BatchNorm2d
        else:
            raise TypeError('Unsupported normalization type: ', cfg.MODEL.MNV3.NORM_FUNC)

        (stem_block, blocks), out_indices, return_features_num_channels = MNV3[cfg.MODEL.MNV3.MV3_TYPE](norm_layer=norm_func)
        self.conv_stem, self.bn1, self.act1 = stem_block
        self.blocks = blocks
        self.out_indices = out_indices
        self.return_features_num_channels = return_features_num_channels

        efficientnet_init_weights(self)

        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)

    def _freeze_backbone(self, freeze_at):
        if freeze_at <= 0:
            return
        for p in self.conv_setm.parameters():
            p.requires_grad = False
        for p in self.bn1.parameters():
            p.requires_grad = False
        for p in self.act1.parameters():
            p.requires_grad = False

        for i, m in enumerate(self.blocks):
            if i < freeze_at-1:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        # print(x.shape)
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        # print(x.shape)
        ret = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            # print(i, x.shape)
            if i in self.out_indices:
                ret.append(x)
        return ret


def make_blocks(block_args, feature_location='pre_pwl',
             in_chans=3, stem_size=16, channel_multiplier=1.0, output_stride=32, pad_type='',
             act_layer=nn.ReLU, drop_rate=0., drop_connect_rate=0., se_kwargs=None,
             norm_layer=nn.BatchNorm2d, norm_kwargs=None):
    norm_kwargs = norm_kwargs or {}

    # Stem
    stem_size = round_channels(stem_size, channel_multiplier)
    conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
    bn1 = norm_layer(stem_size, **norm_kwargs)
    act1 = act_layer(inplace=True)
    _in_chs = stem_size

    # Middle stages (IR/ER/DS Blocks)
    builder = EfficientNetBuilder(
        channel_multiplier, 8, None, output_stride, pad_type, act_layer, se_kwargs,
        norm_layer, norm_kwargs, drop_connect_rate, feature_location=feature_location, verbose=_DEBUG)
    blocks = nn.Sequential(*builder(_in_chs, block_args))

    if _DEBUG:
        for k, v in builder.features.items():
            print('Feature idx: {}: Name: {}, Channels: {}'.format(k, v['name'], v['num_chs']))

    return [conv_stem, bn1, act1], blocks


def _gen_mobilenet_v3_rw(variant, channel_multiplier=1.0, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=HardSwish,
        se_kwargs=dict(gate_fn=hard_sigmoid, reduce_mid=True, divisor=1),
        **kwargs,
    )
    model_kwargs.pop('num_classes', 0)
    model_kwargs.pop('num_features', 0)
    model_kwargs.pop('head_conv', None)
    model = make_blocks(**model_kwargs)
    return model


def _gen_mobilenet_v3(variant, channel_multiplier=1.0, **kwargs):
    """Creates a MobileNet-V3 model.

    Ref impl: ?
    Paper: https://arxiv.org/abs/1905.02244

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = nn.ReLU
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer = HardSwish
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = nn.ReLU
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16'],
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                # stage 2, 56x56 in
                ['ir_r3_k3_s2_e3_c40'],
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112'],
                # stage 5, 14x14in
                ['ir_r3_k3_s2_e6_c160'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer = HardSwish
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                # stage 2, 56x56 in
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                # stage 5, 14x14in
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=act_layer,
        se_kwargs=dict(act_layer=nn.ReLU, gate_fn=hard_sigmoid, reduce_mid=True, divisor=8),
        **kwargs,
    )
    model_kwargs.pop('num_classes', 0)
    model_kwargs.pop('num_features', 0)
    model_kwargs.pop('head_conv', None)
    model = make_blocks(**model_kwargs)
    return model


@MNV3.register('TF_MNV3_L_075')
def tf_mobilenetv3_large_075(**kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_075', 0.75, **kwargs)
    return_features_num_channels = [24, 32, 88, 720]
    return model, out_indices, return_features_num_channels


@MNV3.register('TF_MNV3_L_100')
def tf_mobilenetv3_large_100(**kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_100', 1.0, **kwargs)
    return_features_num_channels = [24, 40, 112, 960]
    return model, out_indices, return_features_num_channels


@MNV3.register('TF_MNV3_L_MIN_100')
def tf_mobilenetv3_large_minimal_100(**kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_mobilenet_v3('tf_mobilenetv3_large_minimal_100', 1.0, **kwargs)
    return_features_num_channels = [24, 40, 112, 960]
    return model, out_indices, return_features_num_channels


@MNV3.register('TF_MNV3_S_075')
def tf_mobilenetv3_small_075(**kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [0, 1, 3, 5]
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_075', 0.75, **kwargs)
    return_features_num_channels = [16, 24, 40, 432]
    return model, out_indices, return_features_num_channels


@MNV3.register('TF_MNV3_S_100')
def tf_mobilenetv3_small_100(**kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [0, 1, 3, 5]
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_100', 1.0, **kwargs)
    return_features_num_channels = [16, 24, 48, 576]
    return model, out_indices, return_features_num_channels


@MNV3.register('TF_MNV3_S_MIN_100')
def tf_mobilenetv3_small_minimal_100(**kwargs):
    """ MobileNet V3 """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [0, 1, 3, 5]
    model = _gen_mobilenet_v3('tf_mobilenetv3_small_minimal_100', 1.0, **kwargs)
    return_features_num_channels = [16, 24, 48, 576]
    return model, out_indices, return_features_num_channels

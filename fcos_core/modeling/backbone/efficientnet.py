""" PyTorch EfficientNet Family

An implementation of EfficienNet that covers variety of related models with efficient architectures:

* EfficientNet (B0-B8 + Tensorflow pretrained AutoAug/RandAug/AdvProp weight ports)
  - EfficientNet: Rethinking Model Scaling for CNNs - https://arxiv.org/abs/1905.11946
  - CondConv: Conditionally Parameterized Convolutions for Efficient Inference - https://arxiv.org/abs/1904.04971
  - Adversarial Examples Improve Image Recognition - https://arxiv.org/abs/1911.09665

* MixNet (Small, Medium, and Large)
  - MixConv: Mixed Depthwise Convolutional Kernels - https://arxiv.org/abs/1907.09595

* MNasNet B1, A1 (SE), Small
  - MnasNet: Platform-Aware Neural Architecture Search for Mobile - https://arxiv.org/abs/1807.11626

* FBNet-C
  - FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS - https://arxiv.org/abs/1812.03443

* Single-Path NAS Pixel1
  - Single-Path NAS: Designing Hardware-Efficient ConvNets - https://arxiv.org/abs/1904.02877

* And likely more...

Hacked together by Ross Wightman
"""
from .efficientnets.efficientnet_builder import *
from .efficientnets.conv2d_layers import select_conv2d
from .efficientnets.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, \
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

from fcos_core.utils.registry import Registry
from fcos_core.layers import FrozenBatchNorm2d

EFN = Registry()
_DEBUG = False


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


# need to set xml according to the flowing info
default_cfgs = {
    'mnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_b1-74cb7081.pth'),
    'semnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mnasnet_a1-d9418771.pth'),
    'fbnetc_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetc_100-c345b898.pth',
        interpolation='bilinear'),
    'spnasnet_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/spnasnet_100-048bc3f4.pth',
        interpolation='bilinear'),
    'efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b0-d6904d92.pth'),
    'efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b1-533bc792.pth',
        input_size=(3, 240, 240), pool_size=(8, 8)),
    'efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b2_ra-bcdf34b7.pth',
        input_size=(3, 260, 260), pool_size=(9, 9)),
    'tf_efficientnet_b0': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_aa-827b6e33.pth',
        input_size=(3, 224, 224)),
    'tf_efficientnet_b1': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_aa-ea7a6ee0.pth',
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_aa-60c94f97.pth',
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_aa-84b4657e.pth',
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_aa-818f208c.pth',
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ra-9a3e5369.pth',
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_aa-80ba17e4.pth',
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ra-6c08e654.pth',
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b0_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b0_ap-f262efe1.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD, input_size=(3, 224, 224)),
    'tf_efficientnet_b1_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b1_ap-44ef0a3d.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_b2_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b2_ap-2f8e7636.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 260, 260), pool_size=(9, 9), crop_pct=0.890),
    'tf_efficientnet_b3_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b3_ap-aad25bdd.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_b4_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b4_ap-dedb23e6.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 380, 380), pool_size=(12, 12), crop_pct=0.922),
    'tf_efficientnet_b5_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b5_ap-9e82fae8.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 456, 456), pool_size=(15, 15), crop_pct=0.934),
    'tf_efficientnet_b6_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b6_ap-4ffb161f.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 528, 528), pool_size=(17, 17), crop_pct=0.942),
    'tf_efficientnet_b7_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ap-ddb28fec.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 600, 600), pool_size=(19, 19), crop_pct=0.949),
    'tf_efficientnet_b8_ap': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b8_ap-00e169fa.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 672, 672), pool_size=(21, 21), crop_pct=0.954),
    'tf_efficientnet_es': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_es-ca1afbfe.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 224, 224), ),
    'tf_efficientnet_em': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_em-e78cfe58.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'tf_efficientnet_el': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_el-5143854e.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
        input_size=(3, 300, 300), pool_size=(10, 10), crop_pct=0.904),
    'tf_efficientnet_cc_b0_4e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_4e-4362b6b2.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_efficientnet_cc_b0_8e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b0_8e-66184a25.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_efficientnet_cc_b1_8e': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_cc_b1_8e-f7c79ae1.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 240, 240), pool_size=(8, 8), crop_pct=0.882),
    'mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_s-a907afbc.pth'),
    'mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_m-4647fc68.pth'),
    'mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_l-5a9a2ed8.pth'),
    'mixnet_xl': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mixnet_xl-ac5fbe8d.pth'),
    'tf_mixnet_s': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_s-89d3354b.pth'),
    'tf_mixnet_m': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_m-0f4d8805.pth'),
    'tf_mixnet_l': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mixnet_l-6c92e0c8.pth'),
}


class EfficientNet(nn.Module):
    """
    EfficientNet Feature Extractor
    """
    def __init__(self, cfg):
        super(EfficientNet, self).__init__()

        if cfg.MODEL.EFN.NORM_FUNC == 'FrozenBatchNorm2d':
            norm_func = FrozenBatchNorm2d
        elif cfg.MODEL.EFN.NORM_FUNC == 'BatchNorm2d':
            norm_func = nn.BatchNorm2d
        else:
            raise TypeError('Unsupported normalization type: ', cfg.MODEL.EFN.NORM_FUNC)

        (stem_block, self.blocks), out_indices, return_features_num_channels = EFN[cfg.MODEL.EFN.EFN_TYPE](norm_layer=norm_func)
        self.conv_setm, self.bn1, self.act1 = stem_block
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
        print(x.shape)
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        print(x.shape)
        ret = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            print(i, x.shape)
            if i in self.out_indices:
                ret.append(x)
        return ret


def _make_blocks(block_args, feature_location='pre_pwl',
             in_chans=3, stem_size=32, channel_multiplier=1.0, channel_divisor=8, channel_min=None,
             output_stride=32, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_connect_rate=0.,
             se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None):

    norm_kwargs = norm_kwargs or {}

    # Stem
    stem_size = round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
    conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
    bn1 = norm_layer(stem_size, **norm_kwargs)
    act1 = act_layer(inplace=True)
    _in_chs = stem_size

    # Middle stages (IR/ER/DS Blocks)
    builder = EfficientNetBuilder(
        channel_multiplier, channel_divisor, channel_min, output_stride, pad_type, act_layer, se_kwargs,
        norm_layer, norm_kwargs, drop_connect_rate, feature_location=feature_location, verbose=_DEBUG)
    blocks = nn.Sequential(*builder(_in_chs, block_args))

    if _DEBUG:
        for k, v in builder.features.items():
            print('Feature idx: {}: Name: {}, Channels: {}'.format(k, v['name'], v['num_chs']))

    return [conv_stem, bn1, act1], blocks


def make_blocks(model_kwargs):
    model_kwargs.pop('num_classes', 0)
    model_kwargs.pop('num_features', 0)
    model_kwargs.pop('head_conv', None)
    return _make_blocks(**model_kwargs)


def _gen_mnasnet_a1(variant, channel_multiplier=1.0, pretrained=True, **kwargs):
    """Creates a mnasnet-a1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r2_k3_s2_e6_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25'],
        # stage 3, 28x28 in
        ['ir_r4_k3_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    return make_blocks(model_kwargs)


def _gen_mnasnet_b1(variant, channel_multiplier=1.0, pretrained=True, **kwargs):
    """Creates a mnasnet-b1 model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
    Paper: https://arxiv.org/pdf/1807.11626.pdf.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r3_k5_s2_e6_c80'],
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    return make_blocks(model_kwargs)


def _gen_fbnetc(variant, channel_multiplier=1.0, pretrained=True, **kwargs):
    """ FBNet-C

        Paper: https://arxiv.org/abs/1812.03443
        Ref Impl: https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_modeldef.py

        NOTE: the impl above does not relate to the 'C' variant here, that was derived from paper,
        it was used to confirm some building block details
    """
    arch_def = [
        ['ir_r1_k3_s1_e1_c16'],
        ['ir_r1_k3_s2_e6_c24', 'ir_r2_k3_s1_e1_c24'],
        ['ir_r1_k5_s2_e6_c32', 'ir_r1_k5_s1_e3_c32', 'ir_r1_k5_s1_e6_c32', 'ir_r1_k3_s1_e6_c32'],
        ['ir_r1_k5_s2_e6_c64', 'ir_r1_k5_s1_e3_c64', 'ir_r2_k5_s1_e6_c64'],
        ['ir_r3_k5_s1_e6_c112', 'ir_r1_k5_s1_e3_c112'],
        ['ir_r4_k5_s2_e6_c184'],
        ['ir_r1_k3_s1_e6_c352'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=16,
        num_features=1984,  # paper suggests this, but is not 100% clear
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    return make_blocks(model_kwargs)


def _gen_spnasnet(variant, channel_multiplier=1.0, pretrained=True, **kwargs):
    """Creates the Single-Path NAS model from search targeted for Pixel1 phone.

    Paper: https://arxiv.org/abs/1904.02877

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_c16_noskip'],
        # stage 1, 112x112 in
        ['ir_r3_k3_s2_e3_c24'],
        # stage 2, 56x56 in
        ['ir_r1_k5_s2_e6_c40', 'ir_r3_k3_s1_e3_c40'],
        # stage 3, 28x28 in
        ['ir_r1_k5_s2_e6_c80', 'ir_r3_k3_s1_e3_c80'],
        # stage 4, 14x14in
        ['ir_r1_k5_s1_e6_c96', 'ir_r3_k5_s1_e3_c96'],
        # stage 5, 14x14in
        ['ir_r4_k5_s2_e6_c192'],
        # stage 6, 7x7 in
        ['ir_r1_k3_s1_e6_c320_noskip']
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    return make_blocks(model_kwargs)


def _gen_efficientnet(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=True, **kwargs):
    """Creates an EfficientNet model.

    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage

    """
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        act_layer=Swish,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    return make_blocks(model_kwargs)


def _gen_efficientnet_edge(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=True, **kwargs):
    """ Creates an EfficientNet-EdgeTPU model

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu
    """

    arch_def = [
        # NOTE `fc` is present to override a mismatch between stem channels and in chs not
        # present in other models
        ['er_r1_k3_s1_e4_c24_fc24_noskip'],
        ['er_r2_k3_s2_e8_c32'],
        ['er_r4_k3_s2_e8_c48'],
        ['ir_r5_k5_s2_e8_c96'],
        ['ir_r4_k5_s1_e8_c144'],
        ['ir_r2_k5_s2_e8_c192'],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=nn.ReLU,
        **kwargs,
    )
    return make_blocks(model_kwargs)


def _gen_efficientnet_condconv(
        variant, channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=1, pretrained=True, **kwargs):
    """Creates an EfficientNet-CondConv model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv
    """
    arch_def = [
      ['ds_r1_k3_s1_e1_c16_se0.25'],
      ['ir_r2_k3_s2_e6_c24_se0.25'],
      ['ir_r2_k5_s2_e6_c40_se0.25'],
      ['ir_r3_k3_s2_e6_c80_se0.25'],
      ['ir_r3_k5_s1_e6_c112_se0.25_cc4'],
      ['ir_r4_k5_s2_e6_c192_se0.25_cc4'],
      ['ir_r1_k3_s1_e6_c320_se0.25_cc4'],
    ]
    # NOTE unlike official impl, this one uses `cc<x>` option where x is the base number of experts for each stage and
    # the expert_multiplier increases that on a per-model basis as with depth/channel multipliers
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, experts_multiplier=experts_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        act_layer=Swish,
        **kwargs,
    )
    return make_blocks(model_kwargs)


def _gen_mixnet_s(variant, channel_multiplier=1.0, pretrained=True, **kwargs):
    """Creates a MixNet Small model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_a1.1_p1.1_s2_e6_c24', 'ir_r1_k3_a1.1_p1.1_s1_e3_c24'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_p1.1_s2_e6_c80_se0.25_nsw', 'ir_r2_k3.5_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3.5.7_a1.1_p1.1_s1_e6_c120_se0.5_nsw', 'ir_r2_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9.11_s2_e6_c200_se0.5_nsw', 'ir_r2_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        num_features=1536,
        stem_size=16,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    return make_blocks(model_kwargs)


def _gen_mixnet_m(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=True, **kwargs):
    """Creates a MixNet Medium-Large model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet
    Paper: https://arxiv.org/abs/1907.09595
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c24'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3.5.7_a1.1_p1.1_s2_e6_c32', 'ir_r1_k3_a1.1_p1.1_s1_e3_c32'],  # relu
        # stage 2, 56x56 in
        ['ir_r1_k3.5.7.9_s2_e6_c40_se0.5_nsw', 'ir_r3_k3.5_a1.1_p1.1_s1_e6_c40_se0.5_nsw'],  # swish
        # stage 3, 28x28 in
        ['ir_r1_k3.5.7_s2_e6_c80_se0.25_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e6_c80_se0.25_nsw'],  # swish
        # stage 4, 14x14in
        ['ir_r1_k3_s1_e6_c120_se0.5_nsw', 'ir_r3_k3.5.7.9_a1.1_p1.1_s1_e3_c120_se0.5_nsw'],  # swish
        # stage 5, 14x14in
        ['ir_r1_k3.5.7.9_s2_e6_c200_se0.5_nsw', 'ir_r3_k3.5.7.9_p1.1_s1_e6_c200_se0.5_nsw'],  # swish
        # 7x7
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, depth_trunc='round'),
        num_features=1536,
        stem_size=24,
        channel_multiplier=channel_multiplier,
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs
    )
    return make_blocks(model_kwargs)


@EFN.register('MNASNET_100')
def mnasnet_100(pretrained=True, **kwargs):
    """ MNASNet B1, depth multiplier of 1.0. """
    out_indices = [1, 2, 4, 6]
    model = _gen_mnasnet_b1('mnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 96, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('SEMNASNET_100')
def semnasnet_100(pretrained=True, **kwargs):
    """ MNASNet A1 (w/ SE), depth multiplier of 1.0. """
    out_indices = [1, 2, 4, 6]
    model = _gen_mnasnet_a1('semnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('FBNETC_100')
def fbnetc_100(pretrained=True, **kwargs):
    """ FBNet-C """
    if pretrained:
        # pretrained model trained with non-default BN epsilon
        kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    out_indices = [1, 2, 4, 6]
    model = _gen_fbnetc('fbnetc_100', 1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 32, 112, 352]
    return model, out_indices, return_features_num_channels


@EFN.register('SPNASNET_100')
def spnasnet_100(pretrained=True, **kwargs):
    """ Single-Path NAS Pixel1"""
    out_indices = [1, 2, 4, 6]
    model = _gen_spnasnet('spnasnet_100', 1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 96, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('EFN_B0')
def efficientnet_b0(pretrained=True, **kwargs):
    """ EfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_connect_rate should be 0.2
    kwargs['drop_connect_rate'] = 0.2
    kwargs['drop_rate'] = 0.2
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('EFN_B1')
def efficientnet_b1(pretrained=True, **kwargs):
    """ EfficientNet-B1 """
    # NOTE for train, drop_rate should be 0.2, drop_connect_rate should be 0.2
    kwargs['drop_connect_rate'] = 0.2
    kwargs['drop_rate'] = 0.2
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('EFN_B2')
def efficientnet_b2(pretrained=True, **kwargs):
    """ EfficientNet-B2 """
    # NOTE for train, drop_rate should be 0.3, drop_connect_rate should be 0.2
    kwargs['drop_connect_rate'] = 0.2
    kwargs['drop_rate'] = 0.3
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B0')
def tf_efficientnet_b0(pretrained=True, **kwargs):
    """ EfficientNet-B0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b0', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B1')
def tf_efficientnet_b1(pretrained=True, **kwargs):
    """ EfficientNet-B1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b1', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B2')
def tf_efficientnet_b2(pretrained=True, **kwargs):
    """ EfficientNet-B2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b2', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 48, 120, 352]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B3')
def tf_efficientnet_b3(pretrained=True, **kwargs):
    """ EfficientNet-B3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b3', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return_features_num_channels = [32, 48, 136, 384]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B4')
def tf_efficientnet_b4(pretrained=True, **kwargs):
    """ EfficientNet-B4. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b4', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return_features_num_channels = [32, 56, 160, 448]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B5')
def tf_efficientnet_b5(pretrained=True, **kwargs):
    """ EfficientNet-B5. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b5', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return_features_num_channels = [40, 64, 176, 512]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B6')
def tf_efficientnet_b6(pretrained=True, **kwargs):
    """ EfficientNet-B6. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['drop_rate'] = 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b6', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return_features_num_channels = [40, 72, 200, 576]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B7')
def tf_efficientnet_b7(pretrained=True, **kwargs):
    """ EfficientNet-B7. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['drop_rate'] = 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b7', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return_features_num_channels = [48, 80, 224, 640]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B0_AP')
def tf_efficientnet_b0_ap(pretrained=True, **kwargs):
    """ EfficientNet-B0. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b0_ap', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B1_AP')
def tf_efficientnet_b1_ap(pretrained=True, **kwargs):
    """ EfficientNet-B1. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b1_ap', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B2_AP')
def tf_efficientnet_b2_ap(pretrained=True, **kwargs):
    """ EfficientNet-B2. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b2_ap', channel_multiplier=1.1, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 48, 120, 352]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B3_AP')
def tf_efficientnet_b3_ap(pretrained=True, **kwargs):
    """ EfficientNet-B3. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b3_ap', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return_features_num_channels = [32, 48, 136, 384]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B4_AP')
def tf_efficientnet_b4_ap(pretrained=True, **kwargs):
    """ EfficientNet-B4. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b4_ap', channel_multiplier=1.4, depth_multiplier=1.8, pretrained=pretrained, **kwargs)
    return_features_num_channels = [32, 56, 160, 448]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B5_AP')
def tf_efficientnet_b5_ap(pretrained=True, **kwargs):
    """ EfficientNet-B5. Tensorflow compatible variant """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b5_ap', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return_features_num_channels = [40, 64, 176, 512]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B6_AP')
def tf_efficientnet_b6_ap(pretrained=True, **kwargs):
    """ EfficientNet-B6. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['drop_rate'] = 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b6_ap', channel_multiplier=1.8, depth_multiplier=2.6, pretrained=pretrained, **kwargs)
    return_features_num_channels = [40, 72, 200, 576]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B7_AP')
def tf_efficientnet_b7_ap(pretrained=True, **kwargs):
    """ EfficientNet-B7. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['drop_rate'] = 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b7_ap', channel_multiplier=2.0, depth_multiplier=3.1, pretrained=pretrained, **kwargs)
    return_features_num_channels = [48, 80, 224, 640]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_B8_AP')
def tf_efficientnet_b8_ap(pretrained=True, **kwargs):
    """ EfficientNet-B7. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.5
    kwargs['drop_rate'] = 0.5
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet(
        'tf_efficientnet_b8_ap', channel_multiplier=2.2, depth_multiplier=3.6, pretrained=pretrained, **kwargs)
    return_features_num_channels = [56, 88, 248, 704]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_EM')
def tf_efficientnet_em(pretrained=True, **kwargs):
    """ EfficientNet-Edge-Medium. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 5]
    model = _gen_efficientnet_edge(
        'tf_efficientnet_em', channel_multiplier=1.0, depth_multiplier=1.1, pretrained=pretrained, **kwargs)
    return_features_num_channels = [32, 48, 144, 192]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_EL')
def tf_efficientnet_el(pretrained=True, **kwargs):
    """ EfficientNet-Edge-Large. Tensorflow compatible variant  """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 5]
    model = _gen_efficientnet_edge(
        'tf_efficientnet_el', channel_multiplier=1.2, depth_multiplier=1.4, pretrained=pretrained, **kwargs)
    return_features_num_channels = [40, 56, 176, 232]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_CC_B0_4E')
def tf_efficientnet_cc_b0_4e(pretrained=True, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 4 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_connect_rate should be 0.2
    kwargs['drop_connect_rate'] = 0.2
    kwargs['drop_rate'] = 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b0_4e', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_CC_B0_8E')
def tf_efficientnet_cc_b0_8e(pretrained=True, **kwargs):
    """ EfficientNet-CondConv-B0 w/ 8 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_connect_rate should be 0.2
    kwargs['drop_connect_rate'] = 0.2
    kwargs['drop_rate'] = 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b0_8e', channel_multiplier=1.0, depth_multiplier=1.0, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_EFN_CC_B1_8E')
def tf_efficientnet_cc_b1_8e(pretrained=True, **kwargs):
    """ EfficientNet-CondConv-B1 w/ 8 Experts. Tensorflow compatible variant """
    # NOTE for train, drop_rate should be 0.2, drop_connect_rate should be 0.2
    kwargs['drop_connect_rate'] = 0.2
    kwargs['drop_rate'] = 0.2
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 6]
    model = _gen_efficientnet_condconv(
        'tf_efficientnet_cc_b1_8e', channel_multiplier=1.0, depth_multiplier=1.1, experts_multiplier=2,
        pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 112, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('MIXNET_S')
def mixnet_s(pretrained=True, **kwargs):
    """Creates a MixNet Small model.
    """
    out_indices = [1, 2, 4, 5]
    model = _gen_mixnet_s(
        'mixnet_s', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 120, 200]
    return model, out_indices, return_features_num_channels


@EFN.register('MIXNET_M')
def mixnet_m(pretrained=True, **kwargs):
    """Creates a MixNet Medium model.
    """
    out_indices = [1, 2, 4, 5]
    model = _gen_mixnet_m(
        'mixnet_m', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [32, 40, 120, 200]
    return model, out_indices, return_features_num_channels


@EFN.register('MIXNET_L')
def mixnet_l(pretrained=True, **kwargs):
    """Creates a MixNet Large model.
    """
    out_indices = [1, 2, 4, 5]
    model = _gen_mixnet_m(
        'mixnet_l', channel_multiplier=1.3, pretrained=pretrained, **kwargs)
    return_features_num_channels = [40, 56, 160, 264]
    return model, out_indices, return_features_num_channels


@EFN.register('MIXNET_XL')
def mixnet_xl(pretrained=True, **kwargs):
    """Creates a MixNet Extra-Large model.
    Not a paper spec, experimental def by RW w/ depth scaling.
    """
    out_indices = [1, 2, 4, 5]
    model = _gen_mixnet_m(
        'mixnet_xl', channel_multiplier=1.6, depth_multiplier=1.2, pretrained=pretrained, **kwargs)
    return_features_num_channels = [48, 64, 192, 320]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_MIXNET_S')
def tf_mixnet_s(pretrained=True, **kwargs):
    """Creates a MixNet Small model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 5]
    model = _gen_mixnet_s(
        'tf_mixnet_s', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [24, 40, 120, 200]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_MIXNET_M')
def tf_mixnet_m(pretrained=True, **kwargs):
    """Creates a MixNet Medium model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 5]
    model = _gen_mixnet_m(
        'tf_mixnet_m', channel_multiplier=1.0, pretrained=pretrained, **kwargs)
    return_features_num_channels = [32, 40, 120, 200]
    return model, out_indices, return_features_num_channels


@EFN.register('TF_MIXNET_L')
def tf_mixnet_l(pretrained=True, **kwargs):
    """Creates a MixNet Large model. Tensorflow compatible variant
    """
    kwargs['bn_eps'] = BN_EPS_TF_DEFAULT
    kwargs['pad_type'] = 'same'
    out_indices = [1, 2, 4, 5]
    model = _gen_mixnet_m(
        'tf_mixnet_l', channel_multiplier=1.3, pretrained=pretrained, **kwargs)
    return_features_num_channels = [40, 56, 160, 264]
    return model, out_indices, return_features_num_channels

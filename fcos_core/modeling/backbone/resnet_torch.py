import torch
import torch.nn as nn
from fcos_core.utils.flops import count_apply_module, count_no_apply_module
from fcos_core.modeling.utils import fuse_conv_and_bn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.is_fuse = False

    def fuse(self):
        self.conv1 = fuse_conv_and_bn(self.conv1, self.bn1)
        self.conv2 = fuse_conv_and_bn(self.conv2, self.bn2)
        self.is_fuse = True
        del self.bn1, self.bn2

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if not self.is_fuse:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if not self.is_fuse:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def forward_dummy(self, x):

        self.__flops__ = 0

        identity = x

        out, flops = count_apply_module(self.conv1, [x])
        self.__flops__ += flops
        if not self.is_fuse:
            out, flops = count_apply_module(self.bn1, [out])
            self.__flops__ += flops
        out = self.relu(out)
        self.__flops__ += out.numel()

        out, flops = count_apply_module(self.conv2, [out])
        self.__flops__ += flops
        if not self.is_fuse:
            out, flops = count_apply_module(self.bn2, [out])
            self.__flops__ += flops

        if self.downsample is not None:
            identity = x
            for layer in self.downsample:
                identity, flops = count_apply_module(layer, [identity])
                self.__flops__ += flops

        out += identity
        out = self.relu(out)
        self.__flops__ += out.numel()*2

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_fuse = False

    def fuse(self):
        self.conv1 = fuse_conv_and_bn(self.conv1, self.bn1)
        self.conv2 = fuse_conv_and_bn(self.conv2, self.bn2)
        self.conv3 = fuse_conv_and_bn(self.conv3, self.bn3)
        self.is_fuse = True
        del self.bn1, self.bn2, self.bn3

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if not self.is_fuse:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if not self.is_fuse:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if not self.is_fuse:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def forward_dummy(self, x):
        self.__flops__ = 0
        identity = x

        out, flops = count_apply_module(self.conv1, [x])
        self.__flops__ += flops
        if not self.is_fuse:
            out, flops = count_apply_module(self.bn1, [out])
            self.__flops__ += flops
        out = self.relu(out)
        self.__flops__ += out.numel()

        out, flops = count_apply_module(self.conv2, [out])
        self.__flops__ += flops
        if not self.is_fuse:
            out, flops = count_apply_module(self.bn2, [out])
            self.__flops__ += flops
        out = self.relu(out)
        self.__flops__ += out.numel()

        out, flops = count_apply_module(self.conv3, [out])
        self.__flops__ += flops
        if not self.is_fuse:
            out, flops = count_apply_module(self.bn3, [out])
            self.__flops__ += flops

        if self.downsample is not None:
            identity = x
            for layer in self.downsample:
                identity, flops = count_apply_module(layer, [identity])
                self.__flops__ += flops

        out += identity
        out = self.relu(out)
        self.__flops__ += out.numel() * 2

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, freeze_at=2):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.is_fuse = False

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.num_layers = len(layers)

        inplanes = self.inplanes
        for i in range(len(layers)):
            setattr(
                self, 'layer%d' % (i+1),
                self._make_layer(
                    block, inplanes, layers[i],
                    stride=1 if i==0 else 2,
                    dilate=False if i==0 else replace_stride_with_dilation[i-1]
                )
            )
            inplanes *= 2

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                    
        self._freeze_backbone(freeze_at)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                for p in self.conv1.parameters():
                    p.requires_grad = False
                for p in self.bn1.parameters():
                    p.requires_grad = False
                # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
                for p in m.parameters():
                    p.requires_grad = False

    def fuse(self):
        self.conv1 = fuse_conv_and_bn(self.conv1, self.bn1)
        
        for i in range(self.num_layers):
            stage = getattr(self, 'layer%d' % (i + 1))
            for layer in stage:
                layer.fuse()
        
        self.is_fuse = True
        del self.bn1

    def forward(self, x):
        x = self.conv1(x)
        if not self.is_fuse:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        for i in range(self.num_layers):
            x = getattr(self, 'layer%d'%(i+1))(x)
            outputs.append(x)

        return outputs

    def forward_dummy(self, x):

        self.__flops__ = 0

        x, flops = count_apply_module(self.conv1, [x])
        self.__flops__ += flops
        if not self.is_fuse:
            x, flops = count_apply_module(self.bn1, [x])
            self.__flops__ += flops
        x = self.relu(x)
        self.__flops__ += flops

        x_ = self.maxpool(x)
        self.__flops__ += count_no_apply_module(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            x, x_
        )
        x = x_

        outputs = []
        for i in range(self.num_layers):
            stage = getattr(self, 'layer%d' % (i + 1))
            for layer in stage:
                x = layer.forward_dummy(x)
                self.__flops__ += layer.__flops__
            outputs.append(x)

        return outputs


def _resnet(arch, block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(cfg, num_blocks):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        'resnet18', BasicBlock, num_blocks,
        freeze_at=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT
    )


def resnet34(cfg, num_blocks):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        'resnet34', BasicBlock, num_blocks,
        freeze_at=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT
    )


def resnext50_32x4d(cfg, num_blocks):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs = {}
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(
        'resnext50_32x4d', Bottleneck, num_blocks,
        freeze_at=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT
        **kwargs
    )


def resnext101_32x8d(cfg):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs = {}
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet(
        'resnext101_32x8d', Bottleneck, [3, 4, 23],
        freeze_at=cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT,
        **kwargs
    )

import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d
from .utils import ChannelNorm, NoopLayer
from fcos_core.utils.flops import count_apply_module


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS_VISDRONE.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.FCOS_VISDRONE.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS_VISDRONE.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS_VISDRONE.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS_VISDRONE.USE_DCN_IN_TOWER
        head_norm_type = cfg.MODEL.FCOS_VISDRONE.HEAD_NORM_TYPE
        
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS_VISDRONE.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.FCOS_VISDRONE.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(self._make_norm_layer(in_channels, head_norm_type))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(self._make_norm_layer(in_channels, head_norm_type))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS_VISDRONE.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(len(self.fpn_strides))])
    
    def _make_norm_layer(self, in_channels, norm_type):
        # import ipdb; ipdb.set_trace()
        if norm_type == 'GN':
            return nn.GroupNorm(32, in_channels)
        elif norm_type == 'BN':
            return nn.BatchNorm2d(in_channels)
        elif norm_type == 'CN':
            return ChannelNorm(in_channels)
        elif norm_type == 'no':
            return NoopLayer()
        else:
            raise NotImplementedError
    
    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred))
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness

    def forward_dummy(self, x):
        self.__flops__ = 0
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = feature
            for layer in self.cls_tower:
                cls_tower, flops = count_apply_module(layer, [cls_tower])
                self.__flops__ += flops
            box_tower = feature
            for layer in self.bbox_tower:
                box_tower, flops = count_apply_module(layer, [box_tower])
                self.__flops__ += flops

            tmp, flops = count_apply_module(self.cls_logits, [cls_tower])
            logits.append(tmp)
            self.__flops__ += flops

            if self.centerness_on_reg:
                tmp, flops = count_apply_module(self.centerness, [box_tower])
                centerness.append(tmp)
                self.__flops__ += flops
            else:
                tmp, flops = count_apply_module(self.centerness, [cls_tower])
                centerness.append(tmp)
                self.__flops__ += flops

            tmp, flops = count_apply_module(self.bbox_pred, [box_tower])
            bbox_pred = self.scales[l](tmp)
            self.__flops__ += flops + tmp.numel()

            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred))
            else:
                bbox_reg.append(torch.exp(bbox_pred))
            self.__flops__ += bbox_pred.numel()
        return logits, bbox_reg, centerness


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()
        self.head = FCOSHead(cfg, in_channels)
        self.box_selector_test = make_fcos_postprocessor(cfg)
        self.loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.fpn_strides = cfg.MODEL.FCOS_VISDRONE.FPN_STRIDES

    def forward_dummy(self, images, features, targets=None, visualizer=None):
        box_cls, box_regression, centerness = self.head.forward_dummy(features)
        self.__flops__ = self.head.__flops__

    def forward(self, images, features, targets=None, visualizer=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)
        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, targets,
                visualizer
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets, visualizer):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets, visualizer
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
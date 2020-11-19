import math
from .inference import make_fcos_postprocessor, make_fcos_postprocessor_sp
from .loss import make_fcos_loss_evaluator, make_prop_loss_evaluator

from .visualize import *
import spconv
from .utils import dense_to_sparse, expand_mask, ChannelNorm, NoopLayer
from fcos_core.utils.flops import count_apply_module, count_no_apply_module


class ProphetHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ProphetHead, self).__init__()

        self.reduction = cfg.MODEL.TDTS_VISDRONE.PROP_REDUCTION

        tower = []
        for i in range(1):
            tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels//self.reduction,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            tower.append(nn.GroupNorm(32, in_channels//self.reduction))
            tower.append(nn.ReLU())

        self.add_module('tower', nn.Sequential(*tower))
        self.logits = nn.Conv2d(
            in_channels//self.reduction, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.tower]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.TDTS_VISDRONE.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.logits.bias, bias_value)

    def forward(self, x):
        logits = []
        for l, feature in enumerate(x):
            logits.append(self.logits(self.tower(feature)))
        return logits

    def forward_dummy(self, x):
        flops = 0
        logits = []
        for l, feature in enumerate(x):
            for m in self.tower:
                feature, flops0 = count_apply_module(m, [feature])
                flops += flops0
            feature, flops0 = count_apply_module(self.logits, [feature])
            flops += flops0
            logits.append(feature)
        self.__flops__ = flops
        return logits


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, norm_type):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.TDTS_VISDRONE.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.TDTS_VISDRONE.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.TDTS_VISDRONE.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.TDTS_VISDRONE.CENTERNESS_ON_REG
        self.margin = cfg.MODEL.TDTS_VISDRONE.FEATURE_MAP_MARGIN
        self.test_prop_thr = cfg.MODEL.TDTS_VISDRONE.TEST_PROP_THR
        self.test_prop_expand_size = cfg.MODEL.TDTS_VISDRONE.TEST_PROP_EXPAND_SIZE

        conv_func = spconv.SubMConv2d
        sp_algo = spconv.ConvAlgo.Native

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.TDTS_VISDRONE.NUM_CONVS):
            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    indice_key='c0',
                    algo=sp_algo
                )
            )
            cls_tower.append(
                self._make_norm_layer(in_channels, norm_type)
            )
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                    indice_key='c0',
                    algo=sp_algo
                )
            )
            bbox_tower.append(
                self._make_norm_layer(in_channels, norm_type)
            )
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', spconv.SparseSequential(*cls_tower))
        self.add_module('bbox_tower', spconv.SparseSequential(*bbox_tower))
        self.cls_logits = conv_func(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1, bias=True, indice_key='c0', algo=sp_algo
        )
        self.bbox_pred = conv_func(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1, bias=True, indice_key='c0', algo=sp_algo
        )
        self.centerness = conv_func(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1, bias=True, indice_key='c0', algo=sp_algo
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, (nn.Conv2d, spconv.SparseConv2d)):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if hasattr(l, 'bias') and l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.TDTS_VISDRONE.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.Parameter(torch.ones(len(self.fpn_strides)), requires_grad=True)

    def _make_norm_layer(self, in_channels, norm_type):
        if norm_type == 'bn':
            raise NotImplementedError
        elif norm_type == 'gn':
            return nn.GroupNorm(32, in_channels)
        elif norm_type == 'cn':
            return ChannelNorm(in_channels)
        else:
            return NoopLayer()

    def forward(self, x, prop_logits=None, mode='dense'):
        if mode == 'dense':
            return self.forward_dense(x)
        elif mode == 'sparse':
            return self.forward_sparse(x, prop_logits)
        else:
            raise NotImplementedError

    def forward_dummy(self, x, prop_logits=None, mode='dense'):
        if mode == 'dense':
            return self.forward_dense_dummy(x)
        elif mode == 'sparse':
            return self.forward_sparse(x, prop_logits)
        else:
            raise NotImplementedError

    def sparsify(self, x, prop_logits):
        prop_logits = [p.sigmoid() for p in prop_logits]
        masks = [expand_mask((p>self.test_prop_thr).float(), self.test_prop_expand_size,
                             permute=False) for p in prop_logits]
        return dense_to_sparse(x, masks, self.fpn_strides, self.margin)

    def forward_sparse(self, x, prop_logits):
        self.__flops__ = 0
        x_sp, pack_info = self.sparsify(x, prop_logits)
        self.__flops__ += sum([p.numel()+p.numel()*(self.test_prop_expand_size**2)
                           for p in prop_logits])
        levels = pack_info['levels']

        def _conv2d_wrap_forward(m, feature):
            if isinstance(m, spconv.SubMConv2d):
                feature, flops = count_apply_module(m, [feature])
                self.__flops__ += flops
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, ChannelNorm)):
                feature.features = m(feature.features)
                self.__flops__ += 2 * feature.features.numel()
            elif isinstance(m, nn.ReLU):
                feature.features = m(feature.features)
                self.__flops__ += feature.features.numel()
            elif isinstance(m, NoopLayer):
                feature = m(feature)
            else:
                feature.features = m(feature.features)
                print('Unknown operator!')
            return feature

        def _conv2d_wrap_seq_forward(module, feature):
            for l, m in enumerate(module):
                feature = _conv2d_wrap_forward(m, feature)
                # print('layer %d: '%l, feature.features.max())
            return feature

        cls_tower = _conv2d_wrap_seq_forward(self.cls_tower, x_sp)
        box_tower = _conv2d_wrap_seq_forward(self.bbox_tower, x_sp)

        logits = _conv2d_wrap_forward(self.cls_logits, cls_tower)
        if self.centerness_on_reg:
            centerness = _conv2d_wrap_forward(self.centerness, box_tower)
        else:
            centerness = _conv2d_wrap_forward(self.centerness, cls_tower)

        bbox_pred = _conv2d_wrap_forward(self.bbox_pred, box_tower)
        bbox_pred.features = bbox_pred.features * self.scales[levels].view(-1, 1)
        self.__flops__ += bbox_pred.features.numel()
        if self.norm_reg_targets:
            bbox_pred.features = F.relu(bbox_pred.features)
        else:
            bbox_pred.features = torch.exp(bbox_pred.features)
        self.__flops__ += bbox_pred.features.numel()
        return logits, bbox_pred, centerness, pack_info

    def forward_dense(self, x):

        def _conv2d_wrap_forward(m, feature):
            if isinstance(m, spconv.SubMConv2d):
                feature = F.conv2d(feature, m.weight.permute(3, 2, 0, 1).contiguous(),
                                   m.bias, padding=1)
            else:
                feature = m(feature)
            return feature

        def _conv2d_wrap_seq_forward(module, feature):
            for l, m in enumerate(module):
                feature = _conv2d_wrap_forward(m, feature)
                # print('layer %d: ' % l, feature.max())
            return feature

        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = _conv2d_wrap_seq_forward(self.cls_tower, feature)
            box_tower = _conv2d_wrap_seq_forward(self.bbox_tower, feature)

            logits.append(_conv2d_wrap_forward(self.cls_logits, cls_tower))
            if self.centerness_on_reg:
                centerness.append(_conv2d_wrap_forward(self.centerness, box_tower))
            else:
                centerness.append(_conv2d_wrap_forward(self.centerness, cls_tower))

            bbox_pred = self.scales[l] * _conv2d_wrap_forward(self.bbox_pred, box_tower)
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred))
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness, None

    def forward_dense_dummy(self, x):
        self.__flops__ = 0

        def _conv2d_wrap_forward(m, feature):
            if isinstance(m, spconv.SubMConv2d):
                tmp = F.conv2d(feature, m.weight.permute(3, 2, 0, 1).contiguous(),
                                   m.bias, padding=1)
                self.__flops__ += count_no_apply_module(
                    nn.Conv2d(m.in_channels, m.out_channels, m.kernel_size, padding=m.padding),
                    [feature], [tmp]
                )
                feature = tmp
            else:
                feature, flops = count_apply_module(m, [feature])
                self.__flops__ += flops
            return feature

        def _conv2d_wrap_seq_forward(module, feature):
            for l, m in enumerate(module):
                feature = _conv2d_wrap_forward(m, feature)
                # print('layer %d: ' % l, feature.max())
            return feature

        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = _conv2d_wrap_seq_forward(self.cls_tower, feature)
            box_tower = _conv2d_wrap_seq_forward(self.bbox_tower, feature)

            logits.append(_conv2d_wrap_forward(self.cls_logits, cls_tower))
            if self.centerness_on_reg:
                centerness.append(_conv2d_wrap_forward(self.centerness, box_tower))
            else:
                centerness.append(_conv2d_wrap_forward(self.centerness, cls_tower))

            bbox_pred = self.scales[l] * _conv2d_wrap_forward(self.bbox_pred, box_tower)
            self.__flops__ += bbox_pred.numel()
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred))
            else:
                bbox_reg.append(torch.exp(bbox_pred))
            self.__flops__ += bbox_pred.numel()
        return logits, bbox_reg, centerness, None


class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        self.idx = 0

        norm_type = cfg.MODEL.TDTS_VISDRONE.NORM_TYPE
        assert norm_type in ['no', 'cn', 'gn', 'bn']

        self.prop_head = ProphetHead(cfg, in_channels)
        self.head = FCOSHead(cfg, in_channels, norm_type)
        self.box_selector_test = make_fcos_postprocessor(cfg)
        self.box_selector_test_sp = make_fcos_postprocessor_sp(cfg)
        self.loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.prop_loss_evaluator = make_prop_loss_evaluator(cfg)
        self.fpn_strides = cfg.MODEL.TDTS_VISDRONE.FPN_STRIDES
        self.use_stage1_labels = cfg.MODEL.TDTS_VISDRONE.PROP_USE_STAGE1_LABELS
        self.prop_refine_labels = cfg.MODEL.TDTS_VISDRONE.PROP_REFINE_LABELS

        self.nms_thresh = cfg.MODEL.TDTS_VISDRONE.PROP_NMS_THR
        self.apply_nms = cfg.MODEL.TDTS_VISDRONE.PROP_APPLY_NMS
        self.fpn_post_nms_top_n = cfg.MODEL.TDTS_VISDRONE.PROP_FPN_POST_NMS_TOP_N
        self.pre_nms_thresh = cfg.MODEL.TDTS_VISDRONE.PROP_PRE_NMS_THRESH
        self.pre_nms_top_n = cfg.MODEL.TDTS_VISDRONE.PROP_PRE_NMS_TOP_N
        self.prop_start_point = cfg.MODEL.TDTS_VISDRONE.PROP_START_POINT
        self.prop_end_point = cfg.MODEL.TDTS_VISDRONE.PROP_END_POINT
        self.prop_max_weight = cfg.MODEL.TDTS_VISDRONE.PROP_MAX_WEIGHT
        self.test_conv_mode = cfg.MODEL.TDTS_VISDRONE.TEST_CONV_MODE

        self.pixel_mean = torch.tensor(cfg.INPUT.PIXEL_MEAN).to(cfg.MODEL.DEVICE).float().view(1, 3, 1, 1)

    def export_heatmap(self, images, prop_logits, save_path):
        """
        overlay heatmap on images
        """
        images = images.tensors + self.pixel_mean
        heatmaps = [p.sigmoid() for p in prop_logits]

        for i in range(len(prop_logits)):
            heatmap_on_image(images[0], heatmaps[i][0], save_path+'_%d.png'%i, alpha=0.6)

        image = images[0].permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(save_path+'_image.png', np.uint8(image))

    def export_fpn_feats(self, images, features, save_path):
        """
        overlay features on images
        """
        images = images.tensors + self.pixel_mean
        heatmaps = [(f**2).sum(dim=1, keepdim=True) for f in features]

        for i in range(len(features)):
            heatmap_on_image(images[0], heatmaps[i][0], save_path+'_%d.png'%i, alpha=0.6)

        image = images[0].permute(1, 2, 0).cpu().numpy()
        cv2.imwrite(save_path+'_image.png', np.uint8(image))

    def forward_dummy(self, images, features, targets=None, visualizer=None):

        self.__flops__ = 0
        mode = self.test_conv_mode

        if mode == 'sparse':
            prop_logits = self.prop_head.forward_dummy(features)
            self.__flops__ += self.prop_head.__flops__
        else:
            prop_logits = None

        box_cls, box_regression, centerness, pack_info = self.head.forward_dummy(
            features, prop_logits, mode
        )
        self.__flops__ += self.head.__flops__


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
        mode = 'dense' if self.training else self.test_conv_mode
        # self.export_fpn_feats(images, features, 'training_dir/feats/'+str(self.idx))

        if self.training or (not self.training and mode=='sparse'):
            prop_logits = self.prop_head(features)
        else:
            prop_logits = None
        box_cls, box_regression, centerness, pack_info = self.head(features, prop_logits, mode)

        if self.training:
            locations = self.compute_locations(features)
            if visualizer:
                visualizer.images = images.tensors + self.pixel_mean
                visualizer.image_sizes = images.image_sizes
                vis_images_boxes(targets, visualizer)
                vis_fpn_features(features, visualizer)
                vis_objectness(
                    [t.sigmoid() for t in prop_logits],
                    visualizer
                )
                vis_pred_boxes(locations, box_cls, box_regression, centerness,
                               self.box_selector_test, images.image_sizes, visualizer)

            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, prop_logits,
                targets, visualizer,
                images.image_sizes
            )
        else:
            if pack_info is not None:
                # self.export_heatmap(images, prop_logits, 'training_dir/heatmaps/'+str(self.idx))
                # self.idx += 1
                return self._forward_test_sp(
                    pack_info, box_cls, box_regression, centerness, images.image_sizes
                )
            else:
                locations = self.compute_locations(features)
                return self._forward_test(
                    locations, box_cls, box_regression, centerness, images.image_sizes
                )

    def limit_prop_labels_in_stage1_targets(self, prop_labels, stage1_labels, shapes):
        """
        prop_labels: list of n*1*h*w
        stage1_labels: list of n*h*w*1
        """
        bs = prop_labels[0].size(0)
        stage1_labels = [l.reshape(bs, shape[0], shape[1], 1).permute(0, 3, 1, 2)
                         for l, shape in zip(stage1_labels, shapes)]
        prop_labels = [((p>0) & l).int() for p, l in zip(prop_labels, stage1_labels)]
        return prop_labels

    def _forward_train(self, locations, box_cls, box_regression, centerness, prop_logits,
                       targets, visualizer, image_sizes):
        loss_box_cls, loss_box_reg, loss_centerness, num_pos_avg_per_gpu, stage1_labels = \
            self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets, visualizer
        )
        if self.use_stage1_labels:
            shapes = [t.shape[-2:] for t in box_cls]
            bs = box_cls[0].size(0)
            prop_labels = [(l>0).int().reshape(bs, shape[0], shape[1], 1).permute(0, 3, 1, 2)
                             for l, shape in zip(stage1_labels, shapes)]
        else:
            prop_labels = self.generate_prop_labels(
                locations, box_cls, box_regression,
                centerness, image_sizes
            )
            if self.prop_refine_labels:
                # print('before:', sum(t.sum() for t in prop_labels))
                prop_labels = self.limit_prop_labels_in_stage1_targets(
                    prop_labels,
                    stage1_labels,
                    [t.shape[-2:] for t in box_cls]
                )
                # print('after:', sum(t.sum() for t in prop_labels))

        loss_prop = self.prop_loss_evaluator(prop_logits, prop_labels,
                                             num_pos_avg_per_gpu, visualizer)

        if visualizer.iteration < self.prop_start_point:
            weight = 0.0
        else:
            weight = min(
                self.prop_max_weight,
                self.prop_max_weight*(visualizer.iteration-self.prop_start_point)/(self.prop_end_point-self.prop_start_point)
            )

        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_prop": loss_prop * weight
        }

        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, centerness, image_sizes
        )
        return boxes, {}

    def _forward_test_sp(self, pack_info, box_cls, box_regression, centerness, image_sizes):

        locations = pack_info['locations']
        levels = pack_info['levels']
        batch_indices = box_cls.indices[:, 0]

        boxes = self.box_selector_test_sp(
            locations, levels, batch_indices,
            box_cls.features, box_regression.features, centerness.features,
            image_sizes
        )
        return boxes, {}

    def generate_prop_labels(self, locations, box_cls, box_regression, centerness, image_sizes):
        box_cls = [x.detach() for x in box_cls]
        box_regression = [x.detach() for x in box_regression]
        centerness = [x.detach() for x in centerness]

        with torch.no_grad():
            boxlists = self.box_selector_test(
                locations, box_cls, box_regression, centerness, image_sizes,
                nms_thresh=self.nms_thresh, fpn_post_nms_top_n=self.fpn_post_nms_top_n,
                apply_nms=self.apply_nms, pre_nms_thresh=self.pre_nms_thresh, pre_nms_top_n=self.pre_nms_top_n
            )

        levels = []
        indices = []
        batch_indices = []
        for i in range(len(boxlists)):
            levels.append(boxlists[i].get_field('levels'))
            indices.append(boxlists[i].get_field('indices'))
            batch_indices.append(levels[-1].new_empty(len(levels[-1])).fill_(i))
        levels = torch.cat(levels)
        indices = torch.cat(indices)
        batch_indices = torch.cat(batch_indices)

        masks = []
        for l in range(len(self.fpn_strides)):
            n, c, h, w = box_cls[l].shape
            per_indices = indices[levels == l]
            per_batch_indices = batch_indices[levels == l]
            per_linear_indices = per_batch_indices * (h * w) + per_indices
            mask = locations[l].new_zeros(n * h * w, dtype=torch.float32)
            mask[per_linear_indices] = 1.0
            mask = mask.view(n, h, w, 1).permute(0, 3, 1, 2)
            masks.append(mask)

        return masks

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


def build_tdts(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
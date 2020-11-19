import torch
from collections import defaultdict

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        norm_reg_targets,
        fpn_strides,
        bbox_aug_enabled=False,
        clip_to_image=True,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.norm_reg_targets = norm_reg_targets
        self.fpn_strides = fpn_strides
        self.bbox_aug_enabled = bbox_aug_enabled
        self.clip_to_image = clip_to_image

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes, level, pre_nms_thresh, pre_nms_top_n):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        box_cls[:, -1] = 0.0
        candidate_inds = box_cls > pre_nms_thresh
        pre_nms_top_n_ = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n_.clamp(max=pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                per_box_loc = per_box_loc[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist.add_field("levels", per_box_loc.new_empty(len(per_box_loc)).fill_(level))
            boxlist.add_field("indices", per_box_loc)
            if self.clip_to_image:
                boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes,
                nms_thresh=None, fpn_post_nms_top_n=None, apply_nms=True,
                pre_nms_thresh=None, pre_nms_top_n=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        if self.norm_reg_targets:
            box_regression = [reg * self.fpn_strides[l] for l, reg in enumerate(box_regression)]

        sampled_boxes = []
        for level, (loc, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    loc, o, b, c, image_sizes, level,
                    pre_nms_thresh or self.pre_nms_thresh,
                    pre_nms_top_n or self.pre_nms_top_n
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if apply_nms:
            boxlists = self.select_over_all_levels(
                boxlists, self.num_classes,
                nms_thresh or self.nms_thresh,
                fpn_post_nms_top_n or self.fpn_post_nms_top_n,
                box_cls[0].device
            )

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.

    @staticmethod
    def select_over_all_levels(boxlists, num_classes, nms_thresh, fpn_post_nms_top_n, device):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], num_classes, nms_thresh, device=device)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.TDTS_VISDRONE.INFERENCE_TH
    pre_nms_top_n = config.MODEL.TDTS_VISDRONE.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.TDTS_VISDRONE.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    norm_reg_targets = config.MODEL.TDTS_VISDRONE.NORM_REG_TARGETS
    fpn_strides = config.MODEL.TDTS_VISDRONE.FPN_STRIDES
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED
    clip_to_image = config.TEST.CLIP_TO_IMAGE

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.TDTS_VISDRONE.NUM_CLASSES - 1,
        norm_reg_targets=norm_reg_targets,
        fpn_strides=fpn_strides,
        bbox_aug_enabled=bbox_aug_enabled,
        clip_to_image=clip_to_image
    )
    return box_selector


class FCOSSparsePostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        norm_reg_targets,
        fpn_strides,
        bbox_aug_enabled=False,
        clip_to_image=True,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSSparsePostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.norm_reg_targets = norm_reg_targets
        self.fpn_strides = fpn_strides
        self.bbox_aug_enabled = bbox_aug_enabled
        self.clip_to_image = clip_to_image

    def forward_for_single_feature_map_faster(self, locations, box_cls,
                        box_regression, centerness, batch_indices, image_sizes):
        """
        Arguments:
            locations: m*2
            box_cls: sparse tensor, m*C
            box_regression: sparse tensor, m*4
        """
        N = len(image_sizes)

        # multiply the classification scores with centerness scores
        box_cls = box_cls.sigmoid() * centerness.sigmoid()
        box_cls[:, -1] = 0.0  # remove `other` class

        candidate_mask = box_cls > self.pre_nms_thresh
        candidate_nonzeros = candidate_mask.nonzero()
        row_indices = candidate_nonzeros[:, 0]
        labels = candidate_nonzeros[:, 1] + 1
        scores = box_cls[candidate_mask]

        box_regression = box_regression[row_indices]  # offset per box
        locations = locations[row_indices]  # loc per box
        batch_indices = batch_indices[row_indices]  # batch idx per box

        detections = torch.cat(  # coords per box
            [locations - box_regression[:, :2],
             locations + box_regression[:, 2:]],
            dim=1
        )

        idx_sort = batch_indices.argsort()
        detections = detections[idx_sort]
        labels = labels[idx_sort]
        scores = scores[idx_sort]
        keys, counts = torch.unique(batch_indices, sorted=True, return_counts=True)
        imgidx_to_counts = defaultdict(int, dict(zip(keys.tolist(), counts.tolist())))

        st = 0
        results = []
        for i in range(N):

            ed = st + imgidx_to_counts[i]
            per_detections = detections[st:ed]
            per_labels = labels[st:ed]
            per_scores = scores[st:ed]

            if per_labels.numel() > self.pre_nms_top_n:
                per_scores, top_k_indices = \
                    per_scores.topk(self.pre_nms_top_n, sorted=False)
                per_labels = per_labels[top_k_indices]
                per_detections = per_detections[top_k_indices]

            h, w = image_sizes[i]
            boxlist = BoxList(per_detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_labels)
            boxlist.add_field("scores", torch.sqrt(per_scores))
            if self.clip_to_image:
                boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

            st = ed

        return results

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness, batch_indices,
            image_sizes):
        """
        122ms
        Arguments:
            locations: m*2
            box_cls: sparse tensor, m*C
            box_regression: sparse tensor, m*4
        """
        N = len(image_sizes)

        # multiply the classification scores with centerness scores
        box_cls = box_cls.sigmoid() * centerness.sigmoid()
        box_cls[:, -1] = 0.0  # remove `other` class

        results = []
        for i in range(N):

            h, w = image_sizes[i]

            per_batch_idx = batch_indices == i
            per_box_cls = box_cls[per_batch_idx, :]
            per_box_regression = box_regression[per_batch_idx, :]
            per_locations = locations[per_batch_idx, :]

            per_candidate_inds = per_box_cls > self.pre_nms_thresh
            per_pre_nms_top_n = per_candidate_inds.sum().clamp(max=self.pre_nms_top_n).item()
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_box_regression = per_box_regression[per_box_loc]
            per_locations = per_locations[per_box_loc]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n:
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.cat(
                [per_locations-per_box_regression[:, :2],
                 per_locations+per_box_regression[:, 2:]],
                dim=1
            )

            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            if self.clip_to_image:
                boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, levels, batch_indices, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            locations: Tensor, m*2
            box_cls: Tensor, m*C
            box_regression: Tensor, m*4
            centerness: Tensor, m*1
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """

        if self.norm_reg_targets:
            fpn_strides = torch.FloatTensor(self.fpn_strides).to(locations.device)
            box_regression = box_regression * fpn_strides[levels][:, None]

        boxlists = self.forward_for_single_feature_map_faster(
                    locations, box_cls, box_regression, centerness, batch_indices, image_sizes
                )

        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists, box_cls.device)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists, device):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.num_classes, self.nms_thresh, device=device)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor_sp(config):
    pre_nms_thresh = config.MODEL.TDTS_VISDRONE.INFERENCE_TH
    pre_nms_top_n = config.MODEL.TDTS_VISDRONE.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.TDTS_VISDRONE.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    norm_reg_targets = config.MODEL.TDTS_VISDRONE.NORM_REG_TARGETS
    fpn_strides = config.MODEL.TDTS_VISDRONE.FPN_STRIDES
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED
    clip_to_image = config.TEST.CLIP_TO_IMAGE

    box_selector = FCOSSparsePostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.TDTS_VISDRONE.NUM_CLASSES-1,
        norm_reg_targets=norm_reg_targets,
        fpn_strides=fpn_strides,
        bbox_aug_enabled=bbox_aug_enabled,
        clip_to_image=clip_to_image
    )

    return box_selector
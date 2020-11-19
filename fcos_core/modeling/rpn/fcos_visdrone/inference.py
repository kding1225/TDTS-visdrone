import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
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
        bbox_aug_enabled=False
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

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes, level):
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
        
        box_cls[:, :, -1] = 0.0
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

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
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes,
                nms_thresh=None, fpn_post_nms_top_n=None):
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
                    loc, o, b, c, image_sizes, level
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
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
    pre_nms_thresh = config.MODEL.FCOS_VISDRONE.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS_VISDRONE.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS_VISDRONE.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    norm_reg_targets = config.MODEL.FCOS_VISDRONE.NORM_REG_TARGETS
    fpn_strides = config.MODEL.FCOS_VISDRONE.FPN_STRIDES
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS_VISDRONE.NUM_CLASSES - 1,
        norm_reg_targets=norm_reg_targets,
        fpn_strides=fpn_strides,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes


class RetinaNetPostProcessor(RPNPostProcessor):
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
        box_coder=None,
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
        super(RetinaNetPostProcessor, self).__init__(
            pre_nms_thresh, 0, nms_thresh, min_size
        )
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
 
    def add_gt_proposals(self, proposals, targets):
        """
        This function is not used in RetinaNet
        """
        pass

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):  # for each level
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(
                boxlists, self.num_classes,
                self.nms_thresh,
                self.fpn_post_nms_top_n,
                box_regression[0].device
            )

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def forward_for_single_feature_map(
            self, anchors, box_cls, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # put in the same format as anchors
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()
        box_cls[:, :, -1] = 0.0

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        num_anchors = A * H * W

        candidate_inds = box_cls > self.pre_nms_thresh

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, \
        per_candidate_inds, per_anchors in zip(
            box_cls,
            box_regression,
            pre_nms_top_n,
            candidate_inds,
            anchors):

            # Sort and select TopN
            # TODO most of this can be made out of the loop for
            # all images. 
            # TODO:Yang: Not easy to do. Because the numbers of detections are
            # different in each image. Therefore, this part needs to be done
            # per image. 
            per_box_cls = per_box_cls[per_candidate_inds]
 
            per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            per_candidate_nonzeros = \
                    per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

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


def make_retinanet_postprocessor(config, rpn_box_coder, is_train):
    pre_nms_thresh = config.MODEL.RETINA_VISDRONE.INFERENCE_TH
    pre_nms_top_n = config.MODEL.RETINA_VISDRONE.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.RETINA_VISDRONE.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    min_size = 0

    box_selector = RetinaNetPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=min_size,
        num_classes=config.MODEL.RETINA_VISDRONE.NUM_CLASSES-1,
        box_coder=rpn_box_coder,
    )

    return box_selector

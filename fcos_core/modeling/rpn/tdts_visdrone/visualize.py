import inspect
import matplotlib.cm as cm

from fcos_core.utils.tensorboardX import *
from .utils import *


def heatmap_on_image(image, heatmap, save_path, alpha=0.3):
    """
    image: 3*H*W
    heatmap: 1*H*W
    """

    H, W = image.shape[-2:]
    heatmap = (heatmap-heatmap.min())/(heatmap.max()-heatmap.min()+1e-5)
    heatmap = F.interpolate(heatmap[None], size=(H, W), mode='bicubic')[0]  # 1*H*W

    image = image.permute(1, 2, 0).cpu().numpy()
    heatmap = heatmap.permute(1, 2, 0).cpu().numpy()
    heatmap = cm.jet_r(heatmap[:, :, 0])[..., :3] * 255.0
    out = heatmap * alpha + image * (1-alpha)
    cv2.imwrite(save_path, np.uint8(out))


def need_show_decorator(func):
    """
    a decorator judging if we need show at this iteration
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        all_args = inspect.getcallargs(func, *args, **kwargs)
        visualizer = all_args['visualizer']

        if visualizer is None:
            return

        iteration = visualizer.iteration
        period = visualizer.show_period

        if not (iteration and iteration % period == 0):
            return

        return func(*args, **kwargs)

    return wrapper


@need_show_decorator
def vis_image_points(locations, levels, batch_indices, visualizer):
    """
    show samples on image
    """
    if not hasattr(visualizer, 'images'):
        return

    cur_batch_idx = torch.nonzero(batch_indices == 0).squeeze()
    points = locations[cur_batch_idx]
    levels = levels[cur_batch_idx]
    for i in range(3):
        visualizer.vis_image_points(visualizer.images[0], points[levels==i], 'samples_level%d'%i, color='g')


@need_show_decorator
def vis_maskes(maskes, visualizer, tag='maskes'):
    """
    show predicted objectness maps
    maskes: list[tensor]
    """
    for i in range(len(maskes)):
        visualizer.vis_feature_map(
            maskes[i][0].contiguous().detach(),
            '%s/level%d'%(tag, i)
        )

@need_show_decorator
def vis_objectness(objectness_maps, visualizer):
    """
    show true objectness maps
    objectness_maps: list[tensor]
    """
    for i in range(len(objectness_maps)):
        visualizer.vis_feature_map(
            objectness_maps[i][0].detach().float(),
            'prop_prob/level%d'%i
        )


@need_show_decorator
def vis_fpn_features(features, visualizer):
    """
    visualize fpn features P3-P5
    features: list[tensor]
    """
    # P3-P7
    for i, tag in enumerate(['P2', 'P3', 'P4']):
        visualizer.vis_feature_map(features[i][0].detach(), 'feature/'+tag)


@need_show_decorator
def vis_images_boxes(targets, visualizer):
    """
    visualize image with gt boxes overlaid on
    images: ImageList
    targets: BooxList
    """
    if not hasattr(visualizer, 'images'):
        return

    # images and targets
    targets_ = targets[0] if targets is not None else None
    visualizer.vis_image(visualizer.images[0], 'annotation/image', targets_)

@need_show_decorator
def vis_pred_boxes(locations, box_cls, box_regression, centerness, forward_test, image_sizes, visualizer):
    """
    show image with predicted boxes overlaid on
    """
    def select_top_predictions(predictions, threshold=0.3, topk=30):
        """
        Select only topk predictions which have a `score` > threshold,
        and returns the predictions in descending order of score
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        idx = idx[:topk]
        return predictions[idx]

    if not hasattr(visualizer, "images"):
        return

    box_cls = [x.detach() for x in box_cls]
    box_regression = [x.detach() for x in box_regression]
    centerness = [x.detach() for x in centerness]

    with torch.no_grad():
        rets = forward_test(
            locations, box_cls, box_regression, centerness, image_sizes,
            nms_thresh=0.6, fpn_post_nms_top_n=100,
            apply_nms=True
        )

    visualizer.vis_image(
        visualizer.images[0], 'stage2_pred_boxes',
        select_top_predictions(rets[0], threshold=0.25, topk=50)
    )

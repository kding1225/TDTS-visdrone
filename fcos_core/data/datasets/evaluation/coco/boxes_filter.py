from copy import deepcopy
from fcos_core.utils.registry import Registry
from pycocotools.coco import COCO

BOXES_FILTERS = Registry()


@BOXES_FILTERS.register('all')
def filter_all(box, im_w, im_h):
    return True


@BOXES_FILTERS.register('center')
def filter_center(box, im_w, im_h):
    x, y, w, h = box
    cx, cy = x + w/2.0, y + h/2.0
    return im_w*0.25 <= cx <= im_w*0.75 and im_h*0.25 <= cy <= im_h*0.75


@BOXES_FILTERS.register('border')
def filter_border(box, im_w, im_h):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    return cx <= im_w*0.25 or cx >= im_w*0.75 or cy <= im_h*0.25 or cy >= im_h*0.75


@BOXES_FILTERS.register('left')
def filter_left(box, im_w, im_h):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    return cx <= im_w*0.25


@BOXES_FILTERS.register('top')
def filter_top(box, im_w, im_h):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    return cy <= im_h*0.25


@BOXES_FILTERS.register('right')
def filter_right(box, im_w, im_h):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    return cx >= im_w*0.75


@BOXES_FILTERS.register('bottom')
def filter_bottom(box, im_w, im_h):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    return cy >= im_h*0.75


def filter_boxes(coco, filter_opt):
    """
    Args:
    coco: pycocotools/coco/COCO object
    filter_opt: str, filter name
    """
    assert filter_opt in BOXES_FILTERS
    filter = BOXES_FILTERS[filter_opt]

    print('filtering boxes...')

    # in this case, do nothing
    if filter_opt == 'all':
        return coco

    dataset = coco.dataset
    imgs = coco.imgs
    annotations = []
    for i, ann in enumerate(dataset['annotations']):
        image_id = ann['image_id']
        bbox = ann['bbox']
        image_info = imgs[image_id]
        width = image_info['width']
        height = image_info['height']
        if filter(bbox, width, height):
            annotations.append(ann)

    # update coco
    coco_new = COCO()
    coco_new.dataset = deepcopy(dataset)
    coco_new.dataset['annotations'] = annotations
    coco_new.createIndex()

    print('filtering boxes done!')

    return coco_new
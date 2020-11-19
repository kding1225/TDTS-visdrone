# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import numpy as np

import torch
from tqdm import tqdm

from fcos_core.structures.image_list import ImageList
from fcos_core.config import cfg
from fcos_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_ml_nms


def compute_on_dataset(model, data_loader, device, out_dir, timer=None):
    model.eval()

    images = ImageList(
        torch.rand(1, 3, 576, 1024),
        [(576, 1024)]
    )
    for i in range(10):
        model(images.to(device))

    results_dict = {}
    cpu_device = torch.device("cpu")
    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images)
            if timer:
                if device == 'cuda' or device==torch.device('cuda'):
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )

    return results_dict


def compute_on_dataset_flops(model, data_loader, device):
    model.eval()
    outputs = []
    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            output = model(images.to(device))
            outputs.append(output)
    return outputs


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def _export_txtret(txtret_path, predictions, dataset):

    imgs_info = dataset.coco.imgs
    ids = dataset.ids

    for idx, pred in enumerate(predictions):

        idx = ids[idx]

        height, width = imgs_info[idx]['height'], imgs_info[idx]['width']
        pred = pred.resize((width, height))

        file_name = os.path.splitext(imgs_info[idx]['file_name'])[0] + '.txt'
        boxes = pred.bbox.int()
        labels = pred.get_field('labels').int()
        scores = pred.get_field('scores').float()
        xmin, ymin, xmax, ymax = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        width = xmax - xmin
        height = ymax - ymin
        with open(os.path.join(txtret_path, file_name), 'w') as f:
            for i in range(len(labels)):
                f.write("{},{},{},{},{:.4f},{},-1,-1\n".format(
                    xmin[i], ymin[i], width[i], height[i], scores[i], labels[i]
                ))


def show_results(show_path, predictions, root_path, imgs_info):
    from PIL import Image, ImageDraw

    classes = ('__background__', # always index 0
           'ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle',
               'awning-tricycle','bus','motor','others')

    for idx, pred in enumerate(predictions):

        height, width = imgs_info[idx]['height'], imgs_info[idx]['width']
        pred = pred.resize((width, height))

        boxes = pred.bbox.int()
        labels = pred.get_field('labels').int()
        scores = pred.get_field('scores').float()

        img = Image.open(os.path.join(root_path, imgs_info[idx]['file_name']))
        draw = ImageDraw.Draw(img)
        for ix in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[ix]
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            label = int(labels[ix])
            score = float(scores[ix])
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0))
            draw.text([xmin, ymin], classes[label], (255, 0, 0))

        image_name = os.path.splitext(imgs_info[idx]['file_name'])[0] + '.png'
        img.save(os.path.join(show_path, image_name))


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    logger = logging.getLogger("fcos_core.inference")
    device = torch.device(device)  # convert to a torch.device for efficiency
    num_devices = get_world_size()
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    
    if cfg.TEST.USE_CACHE:
        predictions = torch.load(os.path.join(output_folder, "predictions.pth"))
        for i in range(len(predictions)):  # shift to cuda for faster speed
            predictions[i] = predictions[i].to(device)
        logger.info("Cache is used for evaluation, cache path: " + output_folder)
    else:
        logger.info("Do not use cache for evaluation")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        predictions = compute_on_dataset(model, data_loader, device, output_folder, inference_timer)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(dataset),
                num_devices,
            )
        )
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        
        if output_folder:
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

            txtret_path = os.path.join(output_folder, 'txtret')
            if not os.path.exists(txtret_path):
                os.mkdir(txtret_path)

            _export_txtret(txtret_path, predictions, dataset)

            # show_path = os.path.join(output_folder, 'show')
            # if not os.path.exists(show_path):
            #     os.mkdir(show_path)
            # show_results(show_path, predictions, dataset.root, dataset.coco.imgs)

    if not is_main_process():
        return

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
        filter_opt=cfg.TEST.GT_BOXES_FILTER
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def inference_flops(
        model,
        data_loader,
        device="cuda",
):
    device = torch.device(device)  # convert to a torch.device for efficiency
    flops_list = compute_on_dataset_flops(model, data_loader, device)
    return flops_list
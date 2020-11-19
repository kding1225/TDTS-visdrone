# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random
import numpy as np
import math
import cv2

import torch
import torchvision
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if isinstance(target, list):
            target = [t.resize(image.size) for t in target]
        elif target is None:
            return image
        else:
            target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            target = target.transpose(1)
        return image, target


class RandomRotate90(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # image = F.rotate(image, 90)
            image = np.ascontiguousarray(np.rot90(np.array(image), 1))
            image = F.to_pil_image(image)
            target = target.transpose(2)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target


class HSV(object):
    def __init__(self, hgain=0.0138, sgain=0.678, vgain=0.36):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, image, target):
        """
        image: PIL Image, RGB, [0,255]
        """
        image = self.augment_hsv(np.array(image), self.hgain, self.sgain, self.vgain)
        image = F.to_pil_image(image)
        return image, target

    @staticmethod
    def augment_hsv(img, hgain, sgain, vgain):
        """
        img: RGB, [0,255]
        """
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB, dst=img)  # no return needed

        return img  # RGB, [0,255]


class CutOut(object):
    def __init__(self, prob=0.1, iou_thr=0.6):
        self.prob = prob
        self.iou_thr = iou_thr

    def __call__(self, image, labels):
        """
        image: RGB, [0,255]
        """
        if random.random() < self.prob:
            image, keep = self.cutout(np.array(image), labels.bbox.numpy(), self.iou_thr)
            image = F.to_pil_image(image)
            labels = self.remove_boxes(labels, keep)
        return image, labels

    def remove_boxes(self, labels, keep):
        labels.bbox = labels.bbox[keep]
        for k, v in labels.extra_fields.items():
            try:
                labels.extra_fields[k] = v[keep]
            except:
                pass
        return labels

    @staticmethod
    def cutout(image, labels, iou_thr):
        # https://arxiv.org/abs/1708.04552
        # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
        # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509

        orig = np.copy(image)
        h, w = image.shape[:2]

        def bbox_ioa(box1, box2):
            # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
            box2 = box2.transpose()

            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

            # Intersection area
            inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                         (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

            # box2 area
            box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

            # Intersection over box2 area
            return inter_area / box2_area

        # create random masks
        scales = [0.125] * 1 + [0.0625] * 2 + [0.03125] * 4 + [0.015625] * 8 + [0.0078125] * 16  # image size fraction
        keep = np.arange(len(labels))
        for s in scales:
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, :4])  # intersection over area
                labels = labels[ioa < iou_thr]  # remove >60% obscured labels
                keep = keep[ioa < iou_thr]

        if len(keep) == 0:  # in case of all boxes are removed
            keep = np.arange(len(labels))
            image = orig

        return image, keep


class MotionBlur(object):
    """Apply motion blur to the input image using a random-sized kernel.
    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """
    def __init__(self, prob=0.3, blur_limit=(3, 11)):
        self.blur_limit = blur_limit
        self.prob = prob

    def __call__(self, img, labels):
        if random.random() < self.prob:
            kernel = self.get_params()
            img = cv2.filter2D(np.array(img), ddepth=-1, kernel=kernel)
            img = F.to_pil_image(img)
        return img, labels

    def get_params(self):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        if ksize <= 2:
            raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        xs, xe = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if xs == xe:
            ys, ye = random.sample(range(ksize), 2)
        else:
            ys, ye = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)

        # Normalize kernel
        kernel = kernel.astype(np.float32) / np.sum(kernel)
        return kernel
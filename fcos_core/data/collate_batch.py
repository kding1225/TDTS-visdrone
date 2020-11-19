# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from fcos_core.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]

        # count useful area in each batch
        image_sizes = images.image_sizes
        max_size = tuple(max(s) for s in zip(*[siz for siz in image_sizes]))
        max_area = max_size[0] * max_size[1]
        # ratio of useful region in the batch
        RoUR = sum([siz[0]*siz[1] for siz in image_sizes])/(max_area*len(image_sizes))
        images.add_field('RoUR', RoUR)

        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

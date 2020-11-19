# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os, ipdb

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.engine.inference import inference_flops
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.flops import flops_to_string


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # batch size should be 1
    assert cfg.TEST.IMS_PER_BATCH == 1

    save_dir = ""
    logger = setup_logger("fcos_core", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.forward = model.forward_dummy
    model.to(cfg.MODEL.DEVICE)

    if cfg.TEST.FUSE_BN:
        model.fuse()
        print(model)

    num_parameters = sum([param.nelement() for param in model.parameters()])
    logger.info('# parameters totally: '+str(num_parameters))

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT, is_train=False)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    flops_list = []
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        flops_list.extend(inference_flops(
            model,
            data_loader_val,
            device=cfg.MODEL.DEVICE,
        ))
        synchronize()

    mean_flops = sum(flops_list)/len(flops_list)
    print(flops_to_string(mean_flops, precision=4))


if __name__ == "__main__":
    main()

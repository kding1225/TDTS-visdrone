# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
from fcos_core.config import cfg

from fcos_core.utils.comm import get_rank, get_world_size, is_pytorch_1_1_0_or_later


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    meters,
    visualizer
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end  # time to load data
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        if visualizer:
            visualizer.update_iteration(iteration)

        loss_dict = model(images, targets, visualizer=visualizer)

        if visualizer:
            visualizer.digest_events()

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())  # total loss
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        if cfg.MODEL.DEBUG:
            with torch.autograd.detect_anomaly():
                losses.backward()
        else:
            losses.backward()
        # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end  # time to process per batch
        end = time.time()
        meters.update(time=batch_time, data=data_time, RoUR=images.get_field('RoUR'))

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))  # remaining time to finish the training

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if visualizer:
                # add extra info to show
                visualizer.update_curve_values('lr', 'lr', optimizer.param_groups[0]["lr"])
                visualizer.update_curve_values('mem_cost', 'mem', torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

                # visualize curves
                visualizer.vis_curves({**loss_dict_reduced, "loss":losses_reduced}, 'losses')
                visualizer.vis_inner_curves()
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            meters.save(is_main_process=get_rank()==0)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            meters.save(is_main_process=get_rank()==0)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

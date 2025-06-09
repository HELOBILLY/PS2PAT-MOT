from cgi import print_arguments
import datetime
import logging
import time
from apex import amp
import torch.distributed as dist
import torchvision
import numpy as np

from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.utils.comm import get_world_size

from .tensorboard_writer import TensorboardWriter


def do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        logger,
        tensorboard_writer: TensorboardWriter = None
):
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):

        if any(len(target) < 1 for target in targets):
            logger.error(
                "Iteration={iteration + 1} || Image Ids used for training {_} || "
                "targets Length={[len(target) for target in targets]}")
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        ## save training images
        ori_img = 'datasets/DOTA/aug_images/{}_ori.jpg'.format(str(iteration).zfill(6))
        aug_img = 'datasets/DOTA/aug_images/{}_aug.jpg'.format(str(iteration).zfill(6))
        torchvision.utils.save_image(images.tensors[0], ori_img)
        torchvision.utils.save_image(images.tensors[1], aug_img)

        targets = [target.to(device) for target in targets]
        ## save training bbox
        ori_txt = 'datasets/DOTA/aug_txt/{}_ori.txt'.format(str(iteration).zfill(6))
        aug_txt = 'datasets/DOTA/aug_txt/{}_aug.txt'.format(str(iteration).zfill(6))
        ori_result = np.array(targets[0].bbox.cpu())
        aug_result = np.array(targets[1].bbox.cpu())
        np.savetxt(ori_txt, ori_result)
        np.savetxt(aug_txt, aug_result)

        result, loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        # write images / ground truth / evaluation metrics to tensorboard
        tensorboard_writer(iteration, losses_reduced, loss_dict_reduced, images, targets)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if get_world_size() < 2 or dist.get_rank() == 0:
            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

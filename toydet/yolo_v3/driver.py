"""YOLOv3 Driver."""

import logging
from argparse import ArgumentParser
from multiprocessing import cpu_count
from typing import Union

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.utils import manual_seed, setup_logger
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from toydet.transforms import LetterBox, MultiArgsSequential
from toydet.utils import cuda_info, mem_info
from toydet.yolo_v3.model import YOLOv3
from toydet.yolo_v3.utils import yolo_loss
from toydet.yolo_v3.voc import VOCDetection_, collate_fn

manual_seed(666)


def load_datasets(batch_size, split, transforms, overfit=False):
    is_train = split == "train"
    ds = VOCDetection_(image_set=split, transforms=transforms)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train and not overfit,
        collate_fn=collate_fn,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available(),
    )


def update_fn(batch, net, optimizer, device, split):
    is_train = split == "train"
    net.train(is_train)
    net.to(device)
    imgs, targets = batch[0].to(device), batch[1].to(device)
    loss_item = {}
    with torch.set_grad_enabled(is_train):
        out1, out2, out3 = net(imgs)
        loss_xywh_1, loss_conf_1, loss_cls_1 = yolo_loss(
            out1,
            targets,
            net.neck.block1.yolo_layer.stride,
            net.neck.block1.yolo_layer.scaled_anchors,
        )
        loss_xywh_2, loss_conf_2, loss_cls_2 = yolo_loss(
            out2,
            targets,
            net.neck.block2.yolo_layer.stride,
            net.neck.block2.yolo_layer.scaled_anchors,
        )
        loss_xywh_3, loss_conf_3, loss_cls_3 = yolo_loss(
            out3,
            targets,
            net.neck.block3.yolo_layer.stride,
            net.neck.block3.yolo_layer.scaled_anchors,
        )
        losses = (
            loss_xywh_1
            + loss_conf_1
            + loss_cls_1
            + loss_xywh_2
            + loss_conf_2
            + loss_cls_2
            + loss_xywh_3
            + loss_conf_3
            + loss_cls_3
        )
        loss_item["stride"] = (
            net.neck.block1.yolo_layer.stride,
            net.neck.block2.yolo_layer.stride,
            net.neck.block3.yolo_layer.stride,
        )
        loss_item["loss_xywh"] = (
            loss_xywh_1.detach().cpu().item(),
            loss_xywh_2.detach().cpu().item(),
            loss_xywh_3.detach().cpu().item(),
        )
        loss_item["loss_cls"] = (
            loss_cls_1.detach().cpu().item(),
            loss_cls_2.detach().cpu().item(),
            loss_cls_3.detach().cpu().item(),
        )
        loss_item["loss_conf"] = (
            loss_conf_1.detach().cpu().item(),
            loss_conf_2.detach().cpu().item(),
            loss_conf_3.detach().cpu().item(),
        )
        # loss_item["ious"] = (
        #     ious_scores_1.detach().cpu().item(),
        #     ious_scores_2.detach().cpu().item(),
        #     ious_scores_3.detach().cpu().item(),
        # )
        loss_item["losses"] = losses.detach().cpu().item()

    if is_train:
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss_item


def show_metrics(engine, logger, split):
    ptable = PrettyTable(["Metrics", "YOLOLayer 1", "YOLOLayer 2", "YOLOLayer 3"])
    output = engine.state.output
    ptable.add_row(
        ["grid_size", output["stride"][0], output["stride"][1], output["stride"][2]]
    )
    ptable.add_row(
        [
            "loss_xywh",
            output["loss_xywh"][0],
            output["loss_xywh"][1],
            output["loss_xywh"][2],
        ]
    )
    ptable.add_row(
        [
            "loss_cls",
            output["loss_cls"][0],
            output["loss_cls"][1],
            output["loss_cls"][2],
        ]
    )
    ptable.add_row(
        [
            "loss_conf",
            output["loss_conf"][0],
            output["loss_conf"][1],
            output["loss_conf"][2],
        ]
    )
    # ptable.add_row(
    #     [
    #         "ious",
    #         output["ious"][0],
    #         output["ious"][1],
    #         output["ious"][2],
    #     ]
    # )
    logger.info(
        "%s Epoch %i - Iteration %i"
        % (split, engine.state.epoch, engine.state.iteration)
    )
    logger.info("%s Results\n%s" % (split, ptable))
    logger.info("%s Total losses - %f" % (split, output["losses"]))


def run(config):
    net = YOLOv3(config.img_size, config.num_classes)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    device = idist.device()

    if config.verbose:
        config.verbose = logging.INFO
    else:
        config.verbose = logging.WARNING

    logger = setup_logger(level=config.verbose, filepath=config.filepath)
    train_engine = Engine(
        lambda engine, batch: update_fn(batch, net, optimizer, device, "train")
    )
    val_engine = Engine(
        lambda engine, batch: update_fn(batch, net, optimizer, device, "val")
    )

    # fire logging event at the end of specified training batches and val epoch
    train_engine.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_every_train_iter),
        show_metrics,
        logger=logger,
        split="Training",
    )
    val_engine.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.log_every_val_epoch),
        show_metrics,
        logger=logger,
        split="Validation",
    )

    # run val_engine before training and at the end of each training epoch
    if config.sanity_check:
        val_dl = load_datasets(2, "val", config.transforms)
        epoch_len = (
            config.sanity_check
            if isinstance(config.sanity_check, int)
            else int(len(val_dl) * config.sanity_check)
        )
        train_engine.add_event_handler(
            Events.STARTED,
            lambda _: val_engine.run(val_dl, max_epochs=1, epoch_length=epoch_len),
        )

    if config.overfit_batches:
        train_dl = load_datasets(config.batch_size, "train", config.transforms, True)
        logger.info("train_dl loaded with overfit %i batches", config.overfit_batches)
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda _: val_engine.run(
                train_dl, max_epochs=1, epoch_length=config.overfit_batches
            ),
        )
    else:
        # create dataloaders
        train_dl = load_datasets(config.batch_size, "train", config.transforms)
        val_dl = load_datasets(config.batch_size, "val", config.transforms)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, lambda _: val_engine.run(val_dl, max_epochs=1)
        )

    logger.info("Running on %s ...", device)
    logger.info("Configs %s", config)
    if "cuda" in device.type:
        name = cuda_info(logger, device)
    # run the training
    train_engine.run(train_dl, max_epochs=config.max_epochs)
    if "cuda" in device.type:
        mem_info(logger, device, name)


if __name__ == "__main__":
    transforms = MultiArgsSequential(LetterBox(416))

    parser = ArgumentParser(description="YOLOv3 Training script")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--img_size", type=int, default=416, help="image size")
    parser.add_argument("--num_classes", type=int, default=20, help="number of classes")
    parser.add_argument("--max_epochs", type=int, default=10, help="epochs to train")
    parser.add_argument("--verbose", "-v", action="store_true", help="use logging.INFO")
    parser.add_argument(
        "--filepath", type=str, default=None, help="file to write log info"
    )
    parser.add_argument(
        "--log_every_train_iter",
        type=int,
        default=50,
        help="log at every n training iteration completed",
    )
    parser.add_argument(
        "--log_every_val_epoch",
        type=int,
        default=1,
        help="log at every n validation epoch completed",
    )
    # parser.add_argument("--amp", action="store_true", help="use cuda.amp")
    parser.add_argument(
        "--sanity_check",
        type=Union[int, float],
        default=2,
        help="amount of batches to sanity check",
    )
    parser.add_argument(
        "--overfit_batches",
        type=int,
        default=0,
        help="overfit the batches",
    )

    config = parser.parse_args()
    config.transforms = transforms
    run(config)

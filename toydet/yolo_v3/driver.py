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

from toydet.transforms import LetterBox, MultiArgsSequential
from toydet.yolo_v3.model import YOLOv3
from toydet.yolo_v3.utils import yolo_loss
from toydet.yolo_v3.voc import VOCDetection_, collate_fn

manual_seed(666)


def load_datasets(bs, split, transforms):
    is_train = split == "train"
    ds = VOCDetection_(image_set=split, transforms=transforms)
    return idist.auto_dataloader(
        ds,
        batch_size=bs,
        shuffle=is_train,
        collate_fn=collate_fn,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    transforms = MultiArgsSequential(LetterBox(416))
    logger = logging.getLogger(__name__)

    parser = ArgumentParser(description="YOLOv3 Training script")
    parser.add_argument("--bs", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--img_size", type=int, default=416, help="image size")
    parser.add_argument("--num_classes", type=int, default=20, help="number of classes")
    parser.add_argument("--epochs", type=int, default=10, help="epochs to train")
    parser.add_argument("--verbose", "-v", action="store_true", help="use logging.INFO")
    # parser.add_argument("--amp", action="store_true", help="use cuda.amp")
    parser.add_argument(
        "--sanity_check",
        type=Union[int, float],
        default=2,
        help="amount of batches to sanity check",
    )
    # parser.add_argument(
    #     "--overfit_batches",
    #     type=Union[int, float],
    #     default=0,
    #     help="overfit the batches",
    # )

    config = parser.parse_args()
    config.transforms = transforms
    if config.verbose:
        config.verbose = logging.INFO
    else:
        config.verbose = logging.WARNING

    net = idist.auto_model(YOLOv3(config.img_size, config.num_classes))
    optimizer = idist.auto_optim(torch.optim.Adam(net.parameters(), lr=config.lr))

    def step_fn(split, batch):
        is_train = split == "train"
        net.train(is_train)
        imgs, targets = batch[0].to(idist.device()), batch[1].to(idist.device())
        with torch.set_grad_enabled(is_train):
            out1, out2, out3 = net(imgs)
            loss_1, losses_1_ = yolo_loss(
                *out1,
                targets,
                net.neck.block1.yolo_layer.stride,
                net.neck.block1.yolo_layer.scaled_anchors,
            )
            loss_2, losses_2_ = yolo_loss(
                *out2,
                targets,
                net.neck.block2.yolo_layer.stride,
                net.neck.block2.yolo_layer.scaled_anchors,
            )
            loss_3, losses_3_ = yolo_loss(
                *out3,
                targets,
                net.neck.block3.yolo_layer.stride,
                net.neck.block3.yolo_layer.scaled_anchors,
            )
            losses = losses_1_ + losses_2_ + losses_3_
            loss_item = dict(
                "grid_size",
                (
                    net.neck.block1.yolo_layer.stride.detach().cpu().item(),
                    net.neck.block2.yolo_layer.stride.detach().cpu().item(),
                    net.neck.block2.yolo_layer.stride.detach().cpu().item(),
                ),
                "loss_xy",
                (
                    loss_1[0].detach().cpu().item(),
                    loss_2[0].detach().cpu().item(),
                    loss_3[0].detach().cpu().item(),
                ),
                "loss_wh",
                (
                    loss_1[1].detach().cpu().item(),
                    loss_2[1].detach().cpu().item(),
                    loss_3[1].detach().cpu().item(),
                ),
                "loss_cls",
                (
                    loss_1[2].detach().cpu().item(),
                    loss_2[2].detach().cpu().item(),
                    loss_3[2].detach().cpu().item(),
                ),
                "loss_conf",
                (
                    loss_1[3].detach().cpu().item(),
                    loss_2[3].detach().cpu().item(),
                    loss_3[3].detach().cpu().item(),
                ),
            )

        if is_train:
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        return loss_item

    def show_metrics(engine):
        ptable = PrettyTable(["Metrics", "YOLOLayer 1", "YOLOLayer 2", "YOLOLayer 3"])
        output = engine.state.output
        ptable.add_row(
            ["grid_size", output["stride"][0], output["stride"][1], output["stride"][2]]
        )
        ptable.add_row(
            [
                "loss_xy",
                output["loss_xy"][0],
                output["loss_xy"][1],
                output["loss_xy"][2],
            ]
        )
        ptable.add_row(
            [
                "loss_wh",
                output["loss_wh"][0],
                output["loss_wh"][1],
                output["loss_wh"][2],
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
        logger.info(
            "==> [%s] – Epoch %i : Batch %i"
            % (engine.logger.name, engine.state.epoch, engine.state.iteration)
        )
        logger.info(ptable)

    # create Engine instances
    train_engine = Engine(lambda engine, batch: step_fn("train", batch))
    val_engine = Engine(lambda engine, batch: step_fn("val", batch))

    # create dataloaders
    train_dl = load_datasets(config.bs, "train", config.transforms)
    val_dl = load_datasets(config.bs, "val", config.transforms)

    # setup logging info
    train_engine.logger = setup_logger("TRAINING", config.verbose)
    val_engine.logger = setup_logger("VALIDATION", config.verbose)

    # fire logging event at the end of 100 training batches and 1 val epoch
    train_engine.add_event_handler(Events.ITERATION_COMPLETED(every=100), show_metrics)
    val_engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), show_metrics)

    # run val_engine before training and at the end of each training epoch
    if config.sanity_check:
        epoch_len = (
            config.sanity_check
            if isinstance(config.sanity_check, int)
            else int(len(val_dl) * config.sanity_check)
        )
        train_engine.add_event_handler(
            Events.STARTED,
            lambda _: val_engine.run(val_dl, max_epochs=1, epoch_length=epoch_len),
        )
    train_engine.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: val_engine.run(val_dl, max_epochs=1)
    )

    # run the training
    train_engine.run(train_dl, max_epochs=config.epochs)

import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime
from random import randint, randrange
from typing import Union, Sequence, Iterable

import hydra
import ignite.distributed as idist
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from ignite.contrib.handlers import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import Precision, Recall
from ignite.utils import manual_seed, setup_logger
from torch import nn, optim, Tensor
from torchvision.ops import batched_nms, box_convert, nms
from torchvision.transforms.functional import to_pil_image

from toydet.transforms import (
    LetterBox,
    RandomHorizontalFlipWithBBox,
    RandomVerticalFlipWithBBox,
)
from toydet.utils import cuda_info, draw_bounding_boxes, mem_info, params_info
from toydet.yolo_v3.models import YOLOv3
from toydet.yolo_v3.datasets import CLASSES, VOCDetection_, get_dataloader

logger = setup_logger()
device = idist.device()
in_colab = "COLAB_TPU_ADDR" in os.environ
with_torch_launch = "WORLD_SIZE" in os.environ


class SequentialWithDict(nn.Module):
    def __init__(self, transforms: Union[dict, OrderedDict]):
        super().__init__()
        self.transforms = nn.ModuleDict(transforms)

    def forward(self, image, target):
        for _, module in self.transforms.items():
            image, target = module(image, target)

        return image, target


# --------------------------
# train and eval transforms
# --------------------------

transforms_train = SequentialWithDict(
    {
        "LetterBox": LetterBox(416),
        "RandomHorizontalFlipWithBBox": RandomHorizontalFlipWithBBox(),
        "RandomVerticalFlipWithBBox": RandomVerticalFlipWithBBox(),
    }
)
transforms_eval = SequentialWithDict({"LetterBox": LetterBox(416)})


# ------------------------------
# sanity checking evaluation
# ------------------------------


def sanity_check(engine: Engine, dataloader: Iterable, config: DictConfig):
    engine.run(dataloader, max_epochs=1, epoch_length=config.sanity_check)
    # set to None to use `epoch_length`
    engine.state.max_epochs = None


# -----------------
# train function
# -----------------


def train_fn(
    engine: Engine,
    batch: Sequence[Tensor],
    net: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer,
):
    net.train(True)
    img, target = batch[0].to(device), batch[1].to(device)
    losses_list = net(img, target)
    losses = sum(loss["loss/total"] for loss in losses_list)
    loss_item = {
        "epoch": engine.state.epoch,
        "loss/xywh/0": losses_list[0]["loss/xywh"],
        "loss/xywh/1": losses_list[1]["loss/xywh"],
        "loss/xywh/2": losses_list[2]["loss/xywh"],
        "loss/cls/0": losses_list[0]["loss/cls"],
        "loss/cls/1": losses_list[1]["loss/cls"],
        "loss/cls/2": losses_list[2]["loss/cls"],
        "loss/conf/0": losses_list[0]["loss/conf"],
        "loss/conf/1": losses_list[1]["loss/conf"],
        "loss/conf/2": losses_list[2]["loss/conf"],
        "loss/total": losses.detach().cpu().item(),
        # "ious/0": ious_scores_1,
        # "ious/1": ious_scores_2,
        # "ious/2": ious_scores_3,
    }

    losses.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {k: v for k, v in sorted(loss_item.items())}


# -------------------
# evaluate function
# -------------------


@torch.no_grad()
def evaluate_fn(
    engine: Engine,
    batch: Sequence[Tensor],
    net: nn.Module,
    device: torch.device,
    conf_threshold: float = 0.5,
):
    net.eval()
    img, target = batch[0].to(device), batch[1].to(device)
    preds = net(img)
    preds = torch.cat(preds, dim=1)
    conf_mask = (preds[:, 4, :] > conf_threshold).float().unsqueeze(1)
    preds = preds * conf_mask
    # os.makedirs("./predictions", exist_ok=True)
    # for i, pred in enumerate(preds):
    #     pred = pred.t()
    #     boxes = box_convert(pred[:, :4], "cxcywh", "xyxy")
    #     scores, idxs = torch.max(pred[:, 5:], 1)
    #     # idxs = torch.randint(0, 20, (10647,))
    #     keep = batched_nms(boxes, scores, idxs, 0.5)
    #     _, box_idx = torch.max(boxes[keep], 0)
    #     best_cls, _ = torch.max(pred[:, 5:][keep], 1)
    #     labels = [CLASSES[int(label)] for label in best_cls.tolist()]
    #     img_ = draw_bounding_boxes(img[i], boxes[keep][box_idx], labels)
    #     img_.save(f"./predictions/{datetime.now().isoformat()}.png", format="png")
    # pred_bbox = box_convert(pred[:, :4, :].reshape(-1, 4), "cxcywh", "xyxy")
    # ious = box_iou(pred_bbox, target[:, 2:6])
    # best_ious, best_n = torch.max(ious, 0)
    # # print(best_ious)
    # cls = pred[:, 5:, :].reshape(-1, 20)
    # best_cls, best_cls_n = torch.max(cls, 1)
    # labels = [CLASSES[int(label)] for label in best_cls.tolist()]
    # img = draw_bounding_boxes(img.squeeze(0), pred_bbox[best_n], labels)
    # img.show()
    return preds[:, 5:, :], target[:, 1]


# Precision(average=True, device=device).attach(engine_eval, "precision")
# Recall(average=True, device=device).attach(engine_eval, "recall")


# -------------------------
# plot transformed images
# -------------------------


def plot_transformed_imgs(engine, config):
    img, target = engine.state.batch[0], engine.state.batch[1].numpy()
    idx = randrange(len(img))
    img = img[idx]
    mask = target[..., 0] == idx
    target = target[mask]

    if config.wandb and config.plot_img:
        box_data = []
        for t in target:
            class_id = int(t[1].item())
            box_data.append(
                {
                    "position": {
                        "minX": t[2].item() / img.shape[-1],
                        "minY": t[3].item() / img.shape[-2],
                        "maxX": t[4].item() / img.shape[-1],
                        "maxY": t[5].item() / img.shape[-2],
                    },
                    "class_id": class_id,
                    "box_caption": CLASSES[class_id],
                }
            )
        boxes = {
            "transforms": {
                "box_data": box_data,
                "class_labels": {k: v for k, v in enumerate(CLASSES)},
            }
        }
        wandb.log({"transforms": wandb.Image(to_pil_image(img, "RGB"), boxes=boxes)})
    elif not config.wandb and config.plot_img:
        boxes = target[..., 2:]
        labels = [CLASSES[int(label)] for label in target[..., 1].tolist()]
        colors = [
            (randint(0, 200), randint(0, 200), randint(0, 200))
            for _ in range(len(labels))
        ]
        img = draw_bounding_boxes(img, boxes, labels, colors)
        fig_name = datetime.now().isoformat() + ".png"
        img.save(fig_name, format="png")
        logger.info("Transformed image saved as %s", fig_name)


# -----------------------
# log metrics to stdout
# -----------------------


def log_metrics(engine, mode, output):
    logger.info(
        "%s Epoch %i - Iteration %i : %s"
        % (mode, engine.state.epoch, engine.state.iteration, output),
    )


def run(local_rank, config: DictConfig) -> None:
    manual_seed(config.seed)

    # ------------------
    # model, optimizer
    # ------------------
    net = idist.auto_model(YOLOv3(config.net.img_size, config.net.num_classes))
    optimizer = idist.auto_optim(optim.Adam(net.parameters(), lr=config.lr))

    # -----------------------
    # train and eval engine
    # -----------------------
    engine_train = Engine(
        lambda engine, batch: train_fn(engine, batch, net, device, optimizer)
    )
    engine_eval = Engine(lambda engine, batch: evaluate_fn(engine, batch, net, device))

    # pylint: disable=not-callable
    engine_train.add_event_handler(
        Events.ITERATION_STARTED(every=config.log_train),
        plot_transformed_imgs,
        config=config,
    )

    # pylint: disable=not-callable
    log_train_events = Events.ITERATION_COMPLETED(
        every=config.log_train
    ) | Events.ITERATION_COMPLETED(once=1)

    engine_train.add_event_handler(
        log_train_events,
        lambda engine: log_metrics(engine, "Train", engine.state.output),
    )

    engine_eval.add_event_handler(
        # pylint: disable=not-callable
        Events.EPOCH_COMPLETED(every=config.log_eval),
        lambda engine: log_metrics(engine, "Eval", engine.state.metrics),
    )

    if config.wandb:
        # --------------
        # wandb logger
        # --------------
        name = f"bs{config.batch_size}-lr{config.lr}"
        wb_logger = WandBLogger(config=config, name=name, project="yolov3")
        wb_logger.watch(net, log="gradients", log_freq=config.log_train)

        # --------------------------
        # logging training metrics
        # --------------------------
        wb_logger.attach_output_handler(
            engine_train,
            Events.ITERATION_COMPLETED(every=config.log_train),
            tag="train",
            output_transform=lambda output: output,
        )

        # ----------------------------
        # logging evaluation metrics
        # ----------------------------
        wb_logger.attach_output_handler(
            engine_eval,
            Events.EPOCH_COMPLETED(every=config.log_eval),
            tag="eval",
            metric_names="all",
            global_step_transform=lambda *_: engine_train.state.iteration,
        )

    # ---------------------------
    # train and eval dataloader
    # ---------------------------
    if config.sanity_check:  # for sanity checking
        dataloader_eval = get_dataloader(
            VOCDetection_, 2, config.path, "val", transforms_eval
        )
        engine_train.add_event_handler(
            Events.STARTED,
            sanity_check,
            engine=engine_eval,
            dataloader=dataloader_eval,
            config=config,
        )

    if config.overfit_batches:  # for overfitting
        dataloader_train = get_dataloader(
            VOCDetection_,
            config.batch_size,
            config.path,
            "train",
            transforms_eval,
            overfit=True,
        )
        engine_train.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda: engine_eval.run(
                dataloader_train, max_epochs=1, epoch_length=config.overfit_batches
            ),
        )
    else:
        dataloader_train = get_dataloader(
            VOCDetection_, config.batch_size, config.path, "train", transforms_train
        )
        dataloader_eval = get_dataloader(
            VOCDetection_, config.batch_size, config.path, "val", transforms_eval
        )
        epoch_length_eval = (
            config.epoch_length_eval
            if isinstance(config.epoch_length_eval, int)
            else round(len(dataloader_eval) * config.epoch_length_eval)
        )
        engine_train.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda: engine_eval.run(
                dataloader_eval, max_epochs=1, epoch_length=epoch_length_eval
            ),
        )

    epoch_length_train = (
        config.epoch_length_train
        if isinstance(config.epoch_length_train, int)
        else round(len(dataloader_train) * config.epoch_length_train)
    )

    engine_train.run(
        dataloader_train, max_epochs=config.max_epochs, epoch_length=epoch_length_train
    )


@hydra.main(config_path="../configs", config_name="defaults")
def main(config: DictConfig = None) -> None:
    logger.info("Running on %s ...", device)
    logger.info(OmegaConf.to_yaml(config))
    if "cuda" in device.type:
        name = cuda_info(logger, device)

    if in_colab or with_torch_launch:
        backend = idist.backend()
        nproc_per_node = idist.get_nproc_per_node()
    else:
        backend = None
        nproc_per_node = None

    with idist.Parallel(backend=backend, nproc_per_node=nproc_per_node) as parallel:
        parallel.run(run, config)

    if "cuda" in device.type:
        mem_info(logger, device, name)


if __name__ == "__main__":
    main()

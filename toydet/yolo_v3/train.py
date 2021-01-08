import os
from collections import OrderedDict
from datetime import datetime
from random import randint, randrange
from typing import Iterable, Sequence, Union

import hydra
import ignite.distributed as idist
import torch
import wandb
from ignite.contrib.handlers import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import Precision, Recall
from ignite.utils import manual_seed, setup_logger, to_onehot
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn, optim
from torchvision.ops import batched_nms, box_convert, nms
from torchvision.transforms.functional import to_pil_image

from toydet.transforms import (
    LetterBox,
    RandomHorizontalFlipWithBBox,
    RandomVerticalFlipWithBBox,
)
from toydet.utils import cuda_info, draw_bounding_boxes, mem_info, params_info
from toydet.yolo_v3.datasets import CLASSES, VOCDetection_, collate_fn
from toydet.yolo_v3.models import YOLOv3

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
    optimizer: optim.Optimizer,
):
    net.train(True)
    img = batch[0].to(device, non_blocking=True)
    target = batch[1].to(device, non_blocking=True)
    loss_dict, total_loss = net(img, target)
    # losses = sum(loss["loss/total"] for loss in losses_list)
    # loss_dict = {
    #     "epoch": engine.state.epoch,
    #     'loss/xywh'
    # "loss/xywh/0": losses_list[0]["loss/xywh"],
    # "loss/xywh/1": losses_list[1]["loss/xywh"],
    # "loss/xywh/2": losses_list[2]["loss/xywh"],
    # "loss/cls/0": losses_list[0]["loss/cls"],
    # "loss/cls/1": losses_list[1]["loss/cls"],
    # "loss/cls/2": losses_list[2]["loss/cls"],
    # "loss/conf/0": losses_list[0]["loss/conf"],
    # "loss/conf/1": losses_list[1]["loss/conf"],
    # "loss/conf/2": losses_list[2]["loss/conf"],
    # "loss/total": losses.detach().cpu().item(),
    # "ious/0": ious_scores_1,
    # "ious/1": ious_scores_2,
    # "ious/2": ious_scores_3,
    # }
    loss_dict["epoch"] = engine.state.epoch

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {k: v for k, v in sorted(loss_dict.items())}


# -------------------
# evaluate function
# -------------------


@torch.no_grad()
def evaluate_fn(
    engine: Engine,
    batch: Sequence[Tensor],
    net: nn.Module,
    conf_threshold: float = 0.5,
):
    net.eval()
    img = batch[0].to(device, non_blocking=True)
    target = batch[1].to(device, non_blocking=True)
    preds = net(img)
    preds = torch.cat(preds, dim=1)
    conf_mask = (preds[:, :, 4] > conf_threshold).float().unsqueeze(-1)
    preds = preds * conf_mask
    # print(preds[..., 5:].shape)
    # print(preds[..., 5:])
    # exit(1)
    # to compute precision, recall
    # pred must be in shape of [B, C, ...]
    # target must be in shape of [B, ...]
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
    return preds, target


# def output_transform(output):
#     preds, _ = output[0][..., 5:]
#     target = to_onehot(output[1][..., 1].to(torch.int64), 20)
#     print(target)
#     return preds, target


# -------------------------
# plot transformed images
# -------------------------


def plot_transformed_imgs(engine, config):
    img, target = engine.state.batch[0], engine.state.batch[1].numpy()
    idx = randrange(len(img))
    img = img[idx]
    mask = target[..., 0] == idx
    target = target[mask]

    if config.wandb and config.save_img:
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
    elif not config.wandb and config.save_img:
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
    if "cuda" in device.type:
        logger.info(
            "Memory: %s MB", torch.cuda.max_memory_allocated(device) / 1024.0 * 1024.0
        )


def run(local_rank: int, config: DictConfig) -> None:
    manual_seed(config.seed)

    # ------------------
    # model, optimizer
    # ------------------
    net = idist.auto_model(YOLOv3(config.net.img_size, config.net.num_classes))
    optimizer = idist.auto_optim(optim.Adam(net.parameters(), lr=config.lr))
    params_info(logger, net)

    # -----------------------------------------
    # Events for logging train and eval info
    # -----------------------------------------
    # pylint: disable=not-callable
    log_train_events = Events.ITERATION_COMPLETED(
        lambda _, event: event % config.log_train == 0 or event == 1
    )
    log_eval_events = Events.EPOCH_COMPLETED(every=config.log_eval)

    # -----------------------
    # train and eval engine
    # -----------------------
    engine_train = Engine(lambda engine, batch: train_fn(engine, batch, net, optimizer))
    engine_eval = Engine(lambda engine, batch: evaluate_fn(engine, batch, net))

    # Precision(output_transform=output_transform, average=True, device=device).attach(
    #     engine_eval, "precision"
    # )
    # Recall(output_transform=output_transform, average=True, device=device).attach(
    #     engine_eval, "recall"
    # )

    # pylint: disable=not-callable
    engine_train.add_event_handler(
        Events.ITERATION_STARTED(every=config.log_train),
        plot_transformed_imgs,
        config=config,
    )

    engine_train.add_event_handler(
        log_train_events,
        lambda engine: log_metrics(engine, "Train", engine.state.output),
    )

    engine_eval.add_event_handler(
        log_eval_events,
        lambda engine: log_metrics(engine, "Eval", engine.state.metrics),
    )

    if config.wandb:
        # --------------
        # wandb logger
        # --------------
        name = f"bs{config.batch_size}-lr{config.lr}"
        wb_logger = WandBLogger(
            config=OmegaConf.to_container(config),
            name=name,
            project="yolov3",
            reinit=True,
        )
        wb_logger.watch(net, log="gradients", log_freq=config.log_train)

        # --------------------------
        # logging training metrics
        # --------------------------
        wb_logger.attach_output_handler(
            engine_train,
            log_train_events,
            tag="train",
            output_transform=lambda output: output,
        )

        # ----------------------------
        # logging evaluation metrics
        # ----------------------------
        wb_logger.attach_output_handler(
            engine_eval,
            log_eval_events,
            tag="eval",
            metric_names="all",
            global_step_transform=lambda *_: engine_train.state.iteration,
        )

    # ---------------------------
    # train and eval dataloader
    # ---------------------------
    dataset_eval = VOCDetection_(
        config.path, image_set="val", transforms=transforms_eval
    )
    dataset_train = VOCDetection_(config.path, transforms=transforms_train)

    if config.sanity_check:  # for sanity checking
        dataloader_eval = idist.auto_dataloader(
            dataset_eval,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.j,
            persistent_workers=True,
        )
        engine_train.add_event_handler(
            Events.STARTED,
            sanity_check,
            engine=engine_eval,
            dataloader=dataloader_eval,
            config=config,
        )

    if config.overfit_batches:  # for overfitting
        dataloader_train = idist.auto_dataloader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.j,
            persistent_workers=True,
        )
        engine_train.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda: engine_eval.run(
                dataloader_train, max_epochs=1, epoch_length=config.overfit_batches
            ),
        )
    else:
        dataloader_train = idist.auto_dataloader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.j,
            persistent_workers=True,
        )
        dataloader_eval = idist.auto_dataloader(
            dataset_eval,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.j,
            persistent_workers=True,
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

    if config.wandb:
        wb_logger.finish()


@hydra.main(config_path="../configs", config_name="defaults")
def main(config: DictConfig = None) -> None:
    logger.info("Running on %s ...", device)
    logger.info(OmegaConf.to_yaml(config))
    if "cuda" in device.type:
        name = cuda_info(logger, device)

    if in_colab or with_torch_launch:
        with idist.Parallel(
            idist.backend(), config.nproc_per_node, config.nnodes, config.node_rank
        ) as parallel:
            parallel.run(run, config)
    else:
        run(None, config)

    if "cuda" in device.type:
        mem_info(logger, device, name)


if __name__ == "__main__":
    main()

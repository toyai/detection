import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from random import randint, randrange
from typing import Iterable, Sequence, Union
from pprint import pformat
import ignite.distributed as idist
import torch
import wandb
from ignite.contrib.handlers import WandBLogger
from ignite.engine import Engine, Events
from ignite.metrics import Precision, Recall
from ignite.utils import manual_seed, setup_logger, to_onehot
from torch import Tensor, nn, optim
from torchvision.ops import batched_nms, box_convert, nms
from torchvision.transforms.functional import to_pil_image

from toydet import get_default_parser
from toydet.transforms import (
    LetterBox,
    RandomHorizontalFlipWithBBox,
    RandomVerticalFlipWithBBox,
)
from toydet.utils import (
    cuda_info,
    draw_bounding_boxes,
    mem_info,
    params_info,
    SequentialWithDict,
)
from toydet.yolo_v3.datasets import CLASSES, VOCDetection_, collate_fn
from toydet.yolo_v3.models import YOLOv3

parser = ArgumentParser(
    "YOLOv3 Training and Evaluation Script", parents=[get_default_parser()]
)
parser.add_argument("--img_size", default=416, type=int, help="input image size")
parser.add_argument("--num_classes", default=20, type=int, help="number of classes")
config = parser.parse_args()
logger = setup_logger(format="[%(levelname)s]: %(message)s")
in_colab = "COLAB_TPU_ADDR" in os.environ
with_torch_launch = "WORLD_SIZE" in os.environ
manual_seed(config.seed)

# -----------------------------------------
# Events for logging train and eval info
# -----------------------------------------

log_train_events = Events.ITERATION_COMPLETED(
    lambda _, event: event % config.log_train == 0 or event == 1
)
log_eval_events = Events.EPOCH_COMPLETED(every=config.log_eval)

# -------------------------
# model, optimizer, device
# -------------------------

net = idist.auto_model(YOLOv3(config.img_size, config.num_classes))
optimizer = idist.auto_optim(optim.Adam(net.parameters(), lr=config.lr))
device = idist.device()

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

# -----------------
# train function
# -----------------


def train_fn(engine: Engine, batch: Sequence[Tensor]):
    net.train(True)
    img = batch[0].to(device, non_blocking=True)
    target = batch[1].to(device, non_blocking=True)
    loss_dict, total_loss = net(img, target)

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_dict["epoch"] = engine.state.epoch
    return {k: v for k, v in sorted(loss_dict.items())}


# -------------------
# evaluate function
# -------------------


@torch.no_grad()
def evaluate_fn(engine: Engine, batch: Sequence[Tensor], conf_threshold: float = 0.5):
    net.eval()
    img = batch[0].to(device, non_blocking=True)
    target = batch[1].to(device, non_blocking=True)
    preds = net(img)
    preds = torch.cat(preds, dim=1)
    conf_mask = (preds[..., 4] > conf_threshold).unsqueeze(-1)
    preds = preds * conf_mask
    preds[..., :4] = box_convert(preds[..., :4], "cxcywh", "xyxy")
    pred_cls, _ = torch.max(preds[..., 5:], dim=-1, keepdim=True)
    preds = torch.cat((preds[..., :5], pred_cls), dim=-1)
    zero_mask = preds != 0
    print(zero_mask.shape)
    print(preds[zero_mask].shape)
    # print(preds)
    # num_classes = preds.shape[-1] - 5
    # conf_mask = (preds[..., 4] > conf_threshold).unsqueeze(-1)
    # preds = preds * conf_mask
    # preds[..., :4] = box_convert(preds[..., :4] * net.img_size, "cxcywh", "xyxy")
    # for i, pred in enumerate(preds):
    #     pred_cls, pred_cls_idx = torch.max(pred[..., 5:], dim=-1, keepdim=True)
    #     pred_cls_idx = pred_cls_idx.float()
    #     pred = torch.cat((pred[..., :5], pred_cls, pred_cls_idx), dim=-1)
    #     non_zero_idx = torch.nonzero(pred[..., 4])
    #     pred = pred[non_zero_idx.squeeze()]
    #     classes = torch.unique_consecutive(pred[..., -1])
    #     keep = nms(pred[..., :4], pred[..., 4], 0.5)
    #     print(pred[keep])
    exit(1)
    return preds, target


engine_train = Engine(train_fn)
engine_eval = Engine(evaluate_fn)

# -------------------------
# plot transformed images
# -------------------------


@engine_train.on(Events.ITERATION_STARTED(every=config.log_train), engine_train, config)
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
            "Memory: %s MB", torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
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
        config=config,
        name=name,
        project="yolov3",
        reinit=True,
    )

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

dataset_eval = VOCDetection_(config.path, image_set="val", transforms=transforms_eval)
dataset_train = VOCDetection_(config.path, transforms=transforms_train)

if config.sanity_check:  # for sanity checking
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.j,
    )

    # ------------------------------
    # sanity checking evaluation
    # ------------------------------
    @engine_train.on(Events.STARTED, engine_eval, dataloader_eval, config)
    def sanity_check(engine: Engine, dataloader: Iterable, config: Namespace):
        engine.run(dataloader, max_epochs=1, epoch_length=config.sanity_check)
        # set to None to use `epoch_length`
        engine.state.max_epochs = None


if config.overfit_batches:  # for overfitting
    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.j,
    )
    engine_train.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda: engine_eval.run(
            dataloader_train,
            max_epochs=1,
            epoch_length=config.overfit_batches,
        ),
    )
else:
    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.j,
    )
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.j,
    )
    epoch_length_eval = (
        config.epoch_length_eval
        if isinstance(config.epoch_length_eval, int)
        else round(len(dataloader_eval) * config.epoch_length_eval)
    )
    engine_train.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda: engine_eval.run(
            dataloader_eval,
            max_epochs=1,
            epoch_length=epoch_length_eval,
        ),
    )

epoch_length_train = (
    config.epoch_length_train
    if isinstance(config.epoch_length_train, int)
    else round(len(dataloader_train) * config.epoch_length_train)
)


def run(local_rank):
    engine_train.run(
        dataloader_train,
        max_epochs=config.max_epochs,
        epoch_length=epoch_length_train,
    )
    if config.wandb:
        wb_logger.finish()


if __name__ == "__main__":
    logger.info("Running on %s ...", device)
    logger.info("\n%s", pformat(vars(config)))
    params_info(logger, net)
    if "cuda" in device.type:
        name = cuda_info(logger, device)

    if in_colab or with_torch_launch:
        with idist.Parallel(
            idist.backend(),
            config.nproc_per_node,
            config.nnodes,
            config.node_rank,
        ) as parallel:
            parallel.run(run)
    else:
        run(None)

    if "cuda" in device.type:
        mem_info(logger, device, name)

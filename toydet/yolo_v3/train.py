import logging
from argparse import ArgumentParser
from datetime import datetime
from random import randint, randrange

import ignite.distributed as idist
import wandb
from ignite.contrib.handlers import WandBLogger
from ignite.engine import Engine, Events
from ignite.utils import manual_seed, setup_logger
from torch import optim
from torchvision.transforms.functional import to_pil_image

from toydet.transforms import (
    LetterBox,
    MultiArgsSequential,
    RandomHorizontalFlip_,
    RandomVerticalFlip_,
)
from toydet.utils import cuda_info, draw_bounding_boxes, mem_info
from toydet.yolo_v3 import models
from toydet.yolo_v3.datasets import CLASSES, VOCDetection_, get_dataloader
from toydet.yolo_v3.engine import evaluate_fn, train_fn

manual_seed(666)

parser = ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--model_name", default="yolov3_darknet53_voc", type=str)
parser.add_argument("--max_epochs", default=10, type=int)
parser.add_argument("--verbose", "-v", action="store_true", help="use logging.INFO")
parser.add_argument("--filepath", default=None, type=str, help="write stdout to file")
parser.add_argument("--log_train", type=int, default=50)
parser.add_argument("--log_eval", type=int, default=1)
# parser.add_argument("--amp", action="store_true")
parser.add_argument("--sanity_check", default=1, type=int, help="0 to disable")
parser.add_argument("--overfit_batches", type=int, default=0)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--plot_img", action="store_true")

config = parser.parse_args()

if config.verbose:
    config.verbose = logging.INFO
else:
    config.verbose = logging.WARNING

# logger
logger = setup_logger(
    format="[%(levelname)s] %(message)s", level=config.verbose, filepath=config.filepath
)

# --------------------------
# model, optimizer, device
# --------------------------
device = idist.device()
net = getattr(models, config.model_name)().to(device)
optimizer = optim.Adam(net.parameters(), lr=config.lr)

# -----------------------
# train and eval engine
# -----------------------
engine_train = Engine(lambda engine, batch: train_fn(batch, net, optimizer, device))
engine_eval = Engine(lambda engine, batch: evaluate_fn(batch, net, optimizer, device))

# --------------------------
# train and eval transforms
# --------------------------
transforms_train = MultiArgsSequential(
    LetterBox(416), RandomHorizontalFlip_(0.8), RandomVerticalFlip_(0.8)
)
transforms_eval = MultiArgsSequential(LetterBox(416))

# ---------------------------
# train and eval dataloader
# ---------------------------
if config.sanity_check:  # for sanity checking
    dataloader_eval = get_dataloader(VOCDetection_, 2, "val", transforms_eval)
    engine_train.add_event_handler(
        Events.STARTED,
        lambda: engine_eval.run(dataloader_eval, max_epochs=1, epoch_length=2),
    )
if config.overfit_batches:  # for overfitting
    dataloader_train = get_dataloader(
        VOCDetection_, config.overfit_batches, "train", transforms_train, overfit=True
    )
    engine_train.add_event_handler(
        Events.EPOCH_COMPLETED, lambda: engine_eval.run(dataloader_train, max_epochs=1)
    )
else:
    dataloader_train = get_dataloader(
        VOCDetection_, config.batch_size, "train", transforms_train
    )
    dataloader_eval = get_dataloader(
        VOCDetection_, config.batch_size, "val", transforms_eval
    )
    engine_train.add_event_handler(
        Events.EPOCH_COMPLETED, lambda: engine_eval.run(dataloader_eval, max_epochs=1)
    )


# -------------------------
# plot transformed images
# -------------------------


@engine_train.on(Events.ITERATION_STARTED(every=config.log_train))
def plot_transformed_imgs(engine):
    img, target = engine.state.batch[0], engine.state.batch[1].numpy()
    idx = randrange(config.batch_size)
    img = img[idx]
    mask = target[..., 0] == idx
    target = target[mask]

    if config.wandb and config.plot_img:
        box_data = []
        for t in target:
            box_data.append(
                {
                    "position": {
                        "minX": t[2].item(),
                        "maxX": t[3].item(),
                        "minY": t[4].item(),
                        "maxY": t[5].item(),
                    },
                    "class_id": int(t[1].item()),
                }
            )
        boxes = {
            "transforms": {
                "box_data": box_data,
                "class_labels": {k: v for k, v in enumerate(CLASSES)},
            }
        }
        wandb.log(
            {"transforms/imgs": wandb.Image(to_pil_image(img, "RGB"), boxes=boxes)}
        )
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


if config.wandb:
    # --------------
    # wandb logger
    # --------------
    wb_logger = WandBLogger(config=config, name=config.model_name, project="yolov3")
    wb_logger.watch(net, log="all")

    # --------------------------
    # logging training metrics
    # --------------------------
    wb_logger.attach_output_handler(
        engine_train,
        Events.ITERATION_COMPLETED(every=config.log_train),
        tag="train",
        metric_names="all",
    )

    # ----------------------------
    # logging evaluation metrics
    # ----------------------------
    wb_logger.attach_output_handler(
        engine_eval,
        Events.EPOCH_COMPLETED(every=config.log_eval),
        tag="eval",
        metric_names="all",
    )

logger.info("Running on %s ...", device)
logger.info("Configs %s", config)

if "cuda" in device.type:
    name = cuda_info(logger, device)

# run the training
engine_train.run(dataloader_train, max_epochs=config.max_epochs)

if "cuda" in device.type:
    mem_info(logger, device, name)

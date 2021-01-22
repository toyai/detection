"""Common utils to use with ignite training / evaluation."""
import logging
import os
from argparse import Namespace
from datetime import datetime
from logging import Logger
from random import randint
from typing import Iterable, Optional, Sequence, Tuple, Union

import ignite.distributed as idist
import numpy as np
import torch
from ignite.engine import Engine
from ignite.utils import setup_logger
from PIL import Image, ImageDraw, ImageFont
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger()

# adapted from https://github.com/pytorch/vision/blob/master/torchvision/utils.py#L138
# for allowing ndarray and PIL.Image


def draw_bounding_boxes(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    boxes: Union[torch.Tensor, np.ndarray, Sequence],
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Tuple[int, int, int]] = None,
    width: int = 1,
) -> Image:
    """Draws bounding boxes on given image."""

    if isinstance(image, torch.Tensor):
        image = to_pil_image(image, "RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image, "RGB")

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.to(torch.int64).tolist()
    elif isinstance(boxes, np.ndarray):
        boxes = boxes.astype(np.int64).tolist()

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        color = (randint(0, 200), randint(0, 200), randint(0, 200))
        draw.rectangle(box, outline=color, width=width)

        if labels is not None:
            x_y = box[0] + 1, box[1]
            text_width, text_height = font.getsize(labels[i])
            draw.rectangle(
                (x_y, (x_y[0] + text_width, x_y[1] + text_height)), fill=color
            )
            draw.text(x_y, labels[i], fill="white", font=font)

    return image


def cuda_info(device: torch.device) -> None:
    """Log cuda info about given ``device``.

    Args:
        device (torch.device): current torch.device.
    """
    if "cuda" in device.type:
        devices = torch.cuda.device_count()
        devices = os.getenv(
            "CUDA_VISIBLE_DEVICES", ",".join([str(i) for i in range(devices)])
        )
        prop = torch.cuda.get_device_properties(device=device)
        logger.info("CUDA_VISIBLE_DEVICES - %s\n%s - %s", devices, prop, device)


def params_info(module: Module) -> None:
    """Log Parameters and Gradients of given ``module``.

    Args:
        module (Module): model which to get parameters and gradients.
    """
    params = sum(p.numel() for p in module.parameters())
    gradients = sum(p.numel() for p in module.parameters() if p.requires_grad)
    logger.info("Parameters: %g, Gradients: %g", params, gradients)


def sanity_check(engine: Engine, dataloader: Iterable, config: Namespace) -> None:
    """Sanity checking with eval dataloader.

    Args:
        engine (Engine): instance of ``Engine`` to run sanity check.
        dataloader (Iterable): ``data`` argument of ``engine.run()``.
        config (Namespace): config namespace object.
    """
    logger.info("Sanity checking with %i iterations.", config.sanity_check)
    engine.run(dataloader, max_epochs=1, epoch_length=config.sanity_check)
    # set to None to use `epoch_length` later
    engine.state.max_epochs = None


def log_metrics(engine: Engine, tag: str, device: torch.device) -> None:
    """Log ``engine.state.metrics`` with given ``engine``
    and memory info with given ``device``.

    Args:
        engine (Engine): instance of ``Engine`` which metrics to log.
        tag (str): a string to add at the start of output.
        device (torch.device): current torch.device to log memory info.
    """
    max_epochs = len(str(engine.state.max_epochs))
    max_iters = len(str(engine.state.max_iters))
    metrics_format = "{0} [{1:>{2}}/{3:0{4}d}]: {5}".format(
        tag,
        engine.state.epoch,
        max_epochs,
        engine.state.iteration,
        max_iters,
        engine.state.metrics,
    )
    if "cuda" in device.type:
        metrics_format += " Memory - {:.2f} MB".format(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )
    logger.info(metrics_format)


def get_dataloaders(
    dataset_train: Dataset,
    dataset_eval: Dataset,
    config: Namespace,
) -> Tuple[DataLoader]:
    """Return ``dataloader_train`` and ``dataloader_eval`` at given config.

    Args:
        dataset_train (Dataset): train dataset.
        dataset_eval (Dataset): eval dataset.
        config (Namespace): config namespace object.

    Returns:
        Tuple[DataLoader]: dataloader_train, dataloader_eval
    """
    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=not config.overfit_batches,
        num_workers=config.j,
        collate_fn=getattr(dataset_train, "collate_fn", None),
        drop_last=config.drop_last,
    )
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.j,
        collate_fn=getattr(dataset_eval, "collate_fn", None),
        drop_last=config.drop_last,
    )
    return dataloader_train, dataloader_eval


def setup_logging(optimizer: Optimizer, config: Namespace) -> Tuple[Logger, str]:
    """Setup logger with ``ignite.utils.setup_logger()``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        config (Namespace): config namespace object.

    Returns:
        Tuple[Logger, str]: instance of ``Logger`` and name
    """
    name = f"bs{config.batch_size}-lr{config.lr}-{optimizer.__class__.__name__}"
    now = datetime.now().strftime("%Y%m%d-%X")
    logger_ = setup_logger(
        level=logging.INFO if config.verbose else logging.WARNING,
        format="> %(message)s",
        filepath=config.filepath / f"{name}-{now}.log",
    )

    return logger_, name

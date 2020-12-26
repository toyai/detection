import os
from logging import Logger
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

# adapted from https://github.com/pytorch/vision/blob/master/torchvision/utils.py#L138
# for allowing ndarray and PIL.Image


def draw_bounding_boxes(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    boxes: Union[torch.Tensor, np.ndarray, Sequence],
    labels: Optional[Sequence[str]] = None,
    colors: Optional[Tuple[int, int, int]] = None,
    width: int = 1,
    font: Optional[str] = None,
    font_size: int = 10,
) -> Image:

    if isinstance(image, torch.Tensor):
        image = to_pil_image(image, "RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image, "RGB")

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.to(torch.int64).tolist()
    elif isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()

    # image_ = image.copy()
    draw = ImageDraw.Draw(image)
    font = (
        ImageFont.load_default()
        if font is None
        else ImageFont.truetype(font=font, size=font_size)
    )

    for i, box in enumerate(boxes):
        color = None if colors is None else colors[i]
        draw.rectangle(box, outline=color, width=width)

        if labels is not None:
            xy = box[0] + 1, box[1]
            text_width, text_height = font.getsize(labels[i])
            draw.rectangle((xy, (xy[0] + text_width, xy[1] + text_height)), fill=color)
            draw.text(xy, labels[i], fill="white", font=font)

    return image


def cuda_info(logger: Logger, device: torch.device):
    devices = torch.cuda.device_count()
    devices = os.getenv(
        "CUDA_VISIBLE_DEVICES", ",".join([str(i) for i in range(devices)])
    )
    logger.info("CUDA_VISIBLE_DEVICES - %s", devices)
    prop = torch.cuda.get_device_properties(device=device)
    logger.info("%s - %s" % (prop, device))

    return prop.name


def mem_info(logger: Logger, device: torch.device, name: str):
    MB = 1024.0 * 1024.0
    logger.info("%s allocated %s MB" % (name, torch.cuda.memory_allocated(device) / MB))
    logger.info(
        "%s allocated max %s MB"
        % (name, torch.cuda.max_memory_allocated(device) * 1e-9)
    )
    logger.info("%s reserved %s MB" % (name, torch.cuda.memory_reserved(device) / MB))
    logger.info(
        "%s reserved max %s MB" % (name, torch.cuda.max_memory_reserved(device) / MB)
    )

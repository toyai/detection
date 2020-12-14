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
    font_size: int = 8,
) -> Image:

    if isinstance(image, torch.Tensor):
        image = to_pil_image(image, "RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image, "RGB")

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.to(torch.int64).tolist()
    elif isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()

    image_ = image.copy()
    draw = ImageDraw.Draw(image_)
    font = (
        ImageFont.load_default()
        if font is None
        else ImageFont.truetype(font=font, size=font_size)
    )

    for i, box in enumerate(boxes):
        color = None if colors is None else colors[i]
        draw.rectangle(box, outline=color, width=width)

        if labels is not None:
            draw.text((box[0] + width, box[1]), labels[i], font=font, fill=color)

    return image_

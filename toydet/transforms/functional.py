import numpy as np
import torch
from torchvision import transforms as T


def letter_box(img, target, size):
    """
    Make letter box transform to image and bounding box target.

    Args:
        img (PIL Image): Image to be transformed.
        target (np.ndarray or Tensor): bounding box target to be transformed (xyxy).
        size (int or tuple of int): the size of the transformed image.

    Returns:
        tuple: (image, target)
    """
    old_width, old_height = img.size
    width, height = size

    ratio = min(width / old_width, height / old_height)
    new_width = int(old_width * ratio)
    new_height = int(old_height * ratio)
    img = T.functional.resize(img, (new_height, new_width))

    pad_x = (width - new_width) * 0.5
    pad_y = (height - new_height) * 0.5
    left, right = round(pad_x + 0.1), round(pad_x - 0.1)
    top, bottom = round(pad_y + 0.1), round(pad_y - 0.1)
    padding = (left, top, right, bottom)
    img = T.functional.pad(img, padding, 255 // 2)

    if isinstance(target, torch.Tensor):
        target[..., 0] = torch.round(ratio * target[..., 0]) + left
        target[..., 1] = torch.round(ratio * target[..., 1]) + top
        target[..., 2] = torch.round(ratio * target[..., 2]) + right
        target[..., 3] = torch.round(ratio * target[..., 3]) + bottom
    elif isinstance(target, np.ndarray):
        target[..., 0] = np.rint(ratio * target[..., 0]) + left
        target[..., 1] = np.rint(ratio * target[..., 1]) + top
        target[..., 2] = np.rint(ratio * target[..., 2]) + right
        target[..., 3] = np.rint(ratio * target[..., 3]) + bottom

    return img, target

import random
from typing import Tuple, Union

import numpy as np
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import functional as FT

from toydet.transforms import functional as F


class LetterBox(nn.Module):
    """
    Make letter box transform to image and bounding box target.

    Args:
        size (int or tuple of int): the size of the transformed image.
    """

    def __init__(self, size: Union[int, Tuple[int]]):
        super().__init__()
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

    def forward(self, img: Image.Image, target: Union[np.ndarray, Tensor]):
        """
        Args:
            img (PIL Image): Image to be transformed.
            target (np.ndarray or Tensor): bounding box target to be transformed.

        Returns:
            tuple: (image, target)
        """
        return F.letter_box(img, target, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f"({self.size})"


class MultiArgsSequential(nn.Sequential):
    def forward(self, inputs, target):
        for module in self:
            inputs, target = module(inputs, target)
        return inputs, target


class RandomHorizontalFlip_(nn.Module):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, img, target):
        if random.random() < self.prob:
            width = img.width
            target[..., 0] = width - target[..., 0]
            target[..., 2] = width - target[..., 2]
            return FT.hflip(img), target
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.prob)


class RandomVerticalFlip_(nn.Module):
    def __init__(self, prob: float = 0.5):
        super().__init__()
        self.prob = prob

    def forward(self, img, target):
        if random.random() < self.prob:
            width = img.width
            target[..., 1] = width - target[..., 1]
            target[..., 3] = width - target[..., 3]
            return FT.vflip(img), target
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.prob)

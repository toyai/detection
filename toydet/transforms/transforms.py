from typing import Tuple, Union

import numpy as np
from PIL import Image
from torch import Tensor, nn

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
    def forward(self, inputs, **kwargs):
        for module in self:
            inputs = module(inputs, **kwargs)
        return inputs

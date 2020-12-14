"""VOC Dataset."""

import os
import xml.etree.ElementTree as ET
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor

from toydet.utils import draw_bounding_boxes

# ----
# VOC
# ----
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class VOCDataset(VOCDetection):
    """
    Args:
        root: same as VOCDetection
        year: same as VOCDetection
        image_set: same as VOCDetection
        download: same as VOCDetection
        transforms (Callable): albumentations transforms applied to image and target

    Returns
        img, targets
    """

    def __init__(
        self,
        root: str = os.getcwd(),
        year: str = "2012",
        image_set: str = "train",
        download: bool = False,
        transforms: Optional[Callable] = None,
        batch_size: Optional[int] = 128,
    ):
        super().__init__(root, year, image_set, download, transforms)
        self.transforms = transforms
        self.batch_size = batch_size
        self.index = 0

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List]:
        # img = cv2.imread(self.images[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(self.images[index]).convert("RGB")

        root_ = ET.parse(self.annotations[index]).getroot()
        targets = []
        for obj in root_.iter("object"):
            target = [0]
            bbox = obj.find("bndbox")
            target.append(CLASSES.index(obj.find("name").text))
            for xyxy in ("xmin", "ymin", "xmax", "ymax"):
                target.append(int(bbox.find(xyxy).text))
            targets.append(target)

        targets = np.array(targets, dtype=np.float32)
        if self.transforms:
            bboxes = targets[:, 2:]
            img, bboxes = self.transforms(img, target=bboxes)
            # transformed = self.transforms(image=img, bboxes=targets)
            # img, targets = transformed["image"], transformed["bboxes"]
            # # return `targets` is a list of tuples
            # targets = [list(t) for t in targets]
            targets[:, 2:] = bboxes

        if (self.index + 1) % self.batch_size == 0:
            boxes = targets[:, 2:]
            labels = [CLASSES[int(label)] for label in targets[:, 1].tolist()]
            img_ = draw_bounding_boxes(img, boxes, labels)
            img_.show()

        self.index += 1
        return img, torch.from_numpy(targets)  # pylint: disable=not-callable


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = [ToTensor()(img) for img in imgs]
    imgs = torch.stack(imgs, dim=0)
    for idx, target in enumerate(targets):
        target[:, 0] = idx

    return imgs, torch.cat(targets, dim=0)

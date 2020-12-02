"""
VOC Dataset
"""

import os
import xml.etree.ElementTree as ET
from typing import Callable, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor

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
    ):
        super().__init__(root, year, image_set, download, transforms)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        img = cv2.imread(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        root_ = ET.parse(self.annotations[index]).getroot()
        targets = []
        for obj in root_.iter("object"):
            target = []
            bbox = obj.find("bndbox")
            for xyxy in ("xmin", "ymin", "xmax", "ymax"):
                target.append(int(bbox.find(xyxy).text))
            target.append(obj.find("name").text)
            targets.append(target)

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=targets)
            img, targets = transformed["image"], transformed["bboxes"]
            # return `targets` is a list of tuples
            targets = [list(t) for t in targets]

        for t in targets:
            t[-1] = CLASSES.index(t[-1]) + 1

        return img, targets


def collate_fn(batch):
    imgs, targets = tuple(zip(*batch))
    imgs = [ToTensor()(img) for img in imgs]
    # max_len = max([len(t) for t in targets])
    # new_targets = []
    # for t in targets:
    #     zeros = np.zeros((max_len - len(t), t.shape[1]))
    #     new_targets.append(np.concatenate((t, zeros), axis=0))

    imgs = torch.stack(imgs, dim=0)
    # targets = torch.tensor(new_targets, dtype=torch.float32)

    return imgs, targets


def draw_bbox(img, target):
    undraw_img = np.copy(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    color = np.random.random(3) * 255
    for t in target:
        xmin, ymin, xmax, ymax, class_id = t
        xmin, ymin, xmax, ymax = map(int, t[:-1])
        name = CLASSES[class_id]
        w, h = cv2.getTextSize(
            text=name, fontFace=font, fontScale=font_scale, thickness=2
        )[0]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=2)
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmin + w, ymin + h),
            color=[c * 0.125 if not 200 > c > 180 else 0 for c in color],
            thickness=cv2.FILLED,
        )
        cv2.putText(
            img,
            text=name,
            org=(xmin, ymin + h),
            fontFace=font,
            fontScale=font_scale,
            color=color,
            thickness=2,
        )
    plt.imshow(img)
    plt.show()

    return undraw_img

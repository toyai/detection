"""VOC Dataset."""

import os
import xml.etree.ElementTree as ET
from multiprocessing import cpu_count
from typing import Callable, List, Optional, Tuple

import ignite.distributed as idist
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VOCDetection
from torchvision.transforms.functional import to_tensor

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


class VOCDetection_(VOCDetection):
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
        super().__init__(root, year, image_set, download)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List]:
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
            bboxes = targets[..., 2:]
            img, bboxes = self.transforms(img, target=bboxes)
            targets[..., 2:] = bboxes

        return img, torch.from_numpy(targets)  # pylint: disable=not-callable


def collate_fn(batch):
    imgs, targets = zip(*batch)
    imgs = [to_tensor(img) for img in imgs]
    imgs = torch.stack(imgs, dim=0)
    for idx, target in enumerate(targets):
        target[:, 0] = idx

    return imgs, torch.cat(targets, dim=0)


def get_dataloader(dataset, batch_size, split, transforms, overfit=False):
    is_train = split == "train"
    ds = dataset(image_set=split, transforms=transforms)
    return idist.auto_dataloader(
        ds,
        batch_size=batch_size,
        shuffle=is_train and not overfit,
        collate_fn=collate_fn,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available(),
    )

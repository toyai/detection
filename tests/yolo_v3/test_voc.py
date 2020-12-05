"""Unitteting everything in yolo_v3/voc.py."""

import os
import unittest

import albumentations as A
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from jiance.yolo_v3.voc import VOCDataset, collate_fn


class TestVOCDataset(unittest.TestCase):
    def test_voc_dataset_without_transforms(self):
        ds = VOCDataset(download=False)
        img, target = next(iter(ds))
        self.assertIsInstance(img, np.ndarray)
        self.assertIsInstance(target, list)
        self.assertTrue(os.path.exists("VOCdevkit"))

    def test_voc_dataset_with_transforms(self):
        img_size = 32
        num_classes = 20
        transforms = A.Compose(
            [A.Resize(img_size, img_size)],
            bbox_params=A.BboxParams(format="pascal_voc"),
        )
        ds = VOCDataset(download=False, transforms=transforms)
        img, targets = next(iter(ds))
        self.assertEqual(img.shape, (img_size, img_size, 3))
        self.assertTrue(
            all(size <= img_size for target in targets for size in target[:4])
        )
        self.assertTrue(
            all(cls_idx <= num_classes for target in targets for cls_idx in target[-1])
        )

    def test_voc_dataloader(self):
        bs = 2
        ds = VOCDataset(download=False)
        dl = DataLoader(ds, batch_size=bs, collate_fn=collate_fn)
        imgs, targets = next(iter(dl))
        self.assertIsInstance(imgs, Tensor)
        self.assertIsInstance(targets, tuple)
        self.assertEqual(len(imgs), bs)
        self.assertEqual(len(targets), bs)


if __name__ == "__main__":
    VOCDataset(download=True)
    unittest.main()

"""Unitteting everything in yolo_v3/voc.py."""

import os
import shutil
from random import randint
from unittest import TestCase, main

import albumentations as A
import numpy as np
from ignite.utils import manual_seed
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from jiance.yolo_v3.voc import CLASSES, VOCDataset, collate_fn

manual_seed(666)


def _create_voc_ds(total_images):
    root = os.path.join(os.getcwd(), "VOCdevkit/VOC2012")
    shutil.rmtree(root)
    annotations = "Annotations"
    jpeg = "JPEGImages"
    img_set = "ImageSets/Main"
    for path in (annotations, jpeg, img_set):
        full_path = os.path.join(root, path)
        os.makedirs(full_path, exist_ok=True)

    for _ in range(total_images):
        name = str(randint(0, 100))
        height, width = randint(0, 1000), randint(0, 1000)
        image = np.random.random((height, width, 3)) * 255
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        xml_template = f"""<annotation>
        <folder>VOC2012</folder>
        <filename>{name}.jpg</filename>
        <source>
            <database>The VOC2008 Database</database>
            <annotation>PASCAL VOC2008</annotation>
            <image>flickr</image>
        </source>
        <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        """
        obj = ""
        for _ in range(randint(0, total_images)):
            class_name = CLASSES[randint(0, 19)]
            obj += f"""<object>
                <name>{class_name}</name>
                <pose>Left</pose>
                <truncated>0</truncated>
                <occluded>1</occluded>
                <bndbox>
                    <xmin>{randint(0, width // 2)}</xmin>
                    <ymin>{randint(0, height // 2)}</ymin>
                    <xmax>{randint(width // 2, width)}</xmax>
                    <ymax>{randint(height // 2, height)}</ymax>
                </bndbox>
                <difficult>0</difficult>
            </object>"""

        xml_template += obj + "</annotation>"
        image.save(os.path.join(root, jpeg, name + ".jpg"))
        with open(os.path.join(root, img_set) + "/train.txt", "a+") as txt:
            txt.write(f"{name}\n")
        with open(os.path.join(root, annotations) + f"/{name}.xml", "a+") as xml:
            xml.write(xml_template)


class TestVOCDataset(TestCase):
    def setUp(self):
        self.img_size = 32
        self.transforms = A.Compose(
            [A.Resize(self.img_size, self.img_size)],
            bbox_params=A.BboxParams(format="pascal_voc"),
        )

    def test_voc_dataset_without_transforms(self):
        ds = VOCDataset(download=False)
        img, target = next(iter(ds))
        self.assertIsInstance(img, np.ndarray)
        self.assertIsInstance(target, list)
        self.assertTrue(os.path.exists("VOCdevkit"))

    def test_voc_dataset_with_transforms(self):
        num_classes = 20
        ds = VOCDataset(download=False, transforms=self.transforms)
        img, targets = next(iter(ds))
        self.assertEqual(img.shape, (self.img_size, self.img_size, 3))
        self.assertTrue(
            all(size <= self.img_size for target in targets for size in target[:4])
        )
        self.assertTrue(
            all(cls_idx <= num_classes for target in targets for cls_idx in target[-1:])
        )

    def test_voc_dataloader(self):
        bs = 2
        ds = VOCDataset(download=False, transforms=self.transforms)
        dl = DataLoader(ds, batch_size=bs, collate_fn=collate_fn)
        for imgs, targets in dl:
            self.assertIsInstance(imgs, Tensor)
            self.assertIsInstance(targets, tuple)
            self.assertTrue(all(isinstance(target, list) for target in targets))
            self.assertEqual(imgs.shape, (bs, 3, self.img_size, self.img_size))
            self.assertEqual(len(targets), bs)


if __name__ == "__main__":
    _create_voc_ds(6)
    main()

"""Unitteting everything in yolo_v3/model.py."""

from unittest import TestCase, main

import torch
from parameterized import parameterized

from toydet.yolo_v3.model import (
    ANCHORS,
    BackBone,
    ConvBN,
    MidBlock,
    Neck,
    ResidualBlock,
    YOLOLayer,
    YOLOv3,
)


class TestYOLO(TestCase):
    def setUp(self):
        self.batch_size = 1
        self.in_channels = 3
        self.img_size = 416
        self.num_classes = 20
        self.x = torch.rand(
            self.batch_size, self.in_channels, self.img_size, self.img_size
        )
        self.out_52 = torch.rand(self.batch_size, 256, 52, 52)
        self.out_26 = torch.rand(self.batch_size, 512, 26, 26)
        self.out_13 = torch.rand(self.batch_size, 1024, 13, 13)

    def test_backbone(self):
        net = BackBone()
        out_52, out_26, out_13 = net(self.x)
        self.assertEqual(out_13.shape, (self.batch_size, 1024, 13, 13))
        self.assertEqual(out_26.shape, (self.batch_size, 512, 26, 26))
        self.assertEqual(out_52.shape, (self.batch_size, 256, 52, 52))

    def test_conv_bn(self):
        net = ConvBN(self.in_channels, 4, kernel_size=3)
        out = net(self.x)
        self.assertEqual(out.shape, (self.batch_size, 4, self.img_size, self.img_size))

    def test_mid_block(self):
        out_channel = 32
        for anchors in ANCHORS:
            with self.subTest(anchors=anchors):
                net = MidBlock(
                    self.in_channels,
                    out_channel,
                    anchors,
                    self.img_size,
                    self.num_classes,
                )
                outs, branch = net(self.x)
                self.assertEqual(
                    outs[0].shape,
                    (
                        self.batch_size,
                        len(anchors),
                        self.img_size,
                        self.img_size,
                        4,
                    ),
                )
                self.assertEqual(
                    outs[1].shape,
                    (
                        self.batch_size,
                        len(anchors),
                        self.img_size,
                        self.img_size,
                        20,
                    ),
                )
                self.assertEqual(
                    outs[2].shape,
                    (
                        self.batch_size,
                        len(anchors),
                        self.img_size,
                        self.img_size,
                        1,
                    ),
                )
                self.assertEqual(
                    branch.shape,
                    (
                        self.batch_size,
                        out_channel // 2,
                        self.img_size,
                        self.img_size,
                    ),
                )

    def test_neck(self):
        net = Neck(self.img_size, self.num_classes)
        out1, out2, out3 = net(self.out_52, self.out_26, self.out_13)
        for idx, out in enumerate((out1, out2, out3)):
            idx = idx + idx if idx > 0 else idx + 1
            self.assertEqual(
                out[0].shape,
                (
                    self.batch_size,
                    len(ANCHORS[2]),
                    self.out_13.size(2) * idx,
                    self.out_13.size(3) * idx,
                    4,
                ),
            )
            self.assertEqual(
                out[1].shape,
                (
                    self.batch_size,
                    len(ANCHORS[1]),
                    self.out_13.size(2) * idx,
                    self.out_13.size(3) * idx,
                    20,
                ),
            )
            self.assertEqual(
                out[2].shape,
                (
                    self.batch_size,
                    len(ANCHORS[0]),
                    self.out_13.size(2) * idx,
                    self.out_13.size(3) * idx,
                    1,
                ),
            )

    def test_residual_block(self):
        net = ResidualBlock(self.in_channels + 1)
        x = torch.rand(self.batch_size, 4, 4, 4)
        out = net(x)
        self.assertEqual(out.shape, (self.batch_size, self.in_channels + 1, 4, 4))

    def test_yolo_layer(self):
        for anchors in ANCHORS:
            x = torch.rand(
                self.batch_size,
                (self.num_classes + 5) * len(anchors),
                self.img_size,
                self.img_size,
            )
            with self.subTest(anchors=anchors):
                net = YOLOLayer(anchors, self.img_size, self.num_classes)
                pred_bbox, pred_cls, pred_conf = net(x)
                self.assertEqual(
                    pred_bbox.shape,
                    (
                        self.batch_size,
                        len(anchors),
                        self.img_size,
                        self.img_size,
                        4,
                    ),
                )
                self.assertEqual(
                    pred_cls.shape,
                    (
                        self.batch_size,
                        len(anchors),
                        self.img_size,
                        self.img_size,
                        20,
                    ),
                )
                self.assertEqual(
                    pred_conf.shape,
                    (
                        self.batch_size,
                        len(anchors),
                        self.img_size,
                        self.img_size,
                        1,
                    ),
                )

    @parameterized.expand([(True,), (False,)])
    def test_yolo_v3(self, mode):
        net = YOLOv3(self.img_size, self.num_classes)
        net.train(mode)
        out1, out2, out3 = net(self.x)
        for idx, out in enumerate((out1, out2, out3)):
            idx = idx + idx if idx > 0 else idx + 1
            self.assertEqual(
                out[0].shape,
                (
                    self.batch_size,
                    len(ANCHORS[2]),
                    self.out_13.size(2) * idx,
                    self.out_13.size(3) * idx,
                    4,
                ),
            )
            self.assertEqual(
                out[1].shape,
                (
                    self.batch_size,
                    len(ANCHORS[1]),
                    self.out_13.size(2) * idx,
                    self.out_13.size(3) * idx,
                    20,
                ),
            )
            self.assertEqual(
                out[2].shape,
                (
                    self.batch_size,
                    len(ANCHORS[0]),
                    self.out_13.size(2) * idx,
                    self.out_13.size(3) * idx,
                    1,
                ),
            )


class TestInvalidYOLO(TestCase):
    def setUp(self):
        self.batch_size = 1
        self.in_channels = 3
        self.img_size = 416
        self.num_classes = 20
        self.x = torch.rand(
            self.batch_size, self.in_channels, self.img_size, self.img_size + 1
        )

    def test_invalid_mid_block(self):
        out_channel = 3
        with self.assertRaisesRegex(
            ValueError, r"out_channels must be divisible by 2."
        ):
            MidBlock(
                self.in_channels,
                out_channel,
                ANCHORS[0],
                self.img_size,
                self.num_classes,
            )

    def test_invalid_residual_block(self):
        with self.assertRaisesRegex(ValueError, r"in_channels must be divisible by 2."):
            ResidualBlock(self.in_channels)

    def test_invalid_yolo_layer(self):
        net = YOLOLayer(ANCHORS[0], self.img_size, self.num_classes)
        with self.assertRaisesRegex(
            ValueError, r"image must have same height and width."
        ):
            net(self.x)

    def test_invalid_yolo_v3(self):
        net = YOLOv3(self.img_size, self.num_classes)
        with self.assertRaisesRegex(
            ValueError, r"image must have same height and width."
        ):
            net(self.x)


if __name__ == "__main__":
    main()

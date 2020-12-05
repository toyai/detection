"""Unitteting everything in yolo_v3/model.py."""

import unittest

import torch

from jiance.yolo_v3.backbone import (
    BackBone,
    ConvBN,
    MidBlock,
    Neck,
    ResidualBlock,
    YOLOLayer,
    YOLOv3,
)

ANCHORS = (
    ((10, 13), (16, 30), (33, 23)),
    ((30, 61), (62, 45), (59, 119)),
    ((116, 90), (156, 198), (373, 326)),
)


class TestYOLO(unittest.TestCase):
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
        self.assertEqual(out_13.size(), torch.Size([self.batch_size, 1024, 13, 13]))
        self.assertEqual(out_26.size(), torch.Size([self.batch_size, 512, 26, 26]))
        self.assertEqual(out_52.size(), torch.Size([self.batch_size, 256, 52, 52]))

    def test_conv_bn(self):
        net = ConvBN(self.in_channels, 4, kernel_size=3)
        out = net(self.x)
        self.assertEqual(
            out.size(), torch.Size([self.batch_size, 4, self.img_size, self.img_size])
        )

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
                    outs.size(),
                    torch.Size(
                        [
                            self.batch_size,
                            len(anchors),
                            self.img_size,
                            self.img_size,
                            self.num_classes + 5,
                        ]
                    ),
                )
                self.assertEqual(
                    branch.size(),
                    torch.Size(
                        [
                            self.batch_size,
                            out_channel // 2,
                            self.img_size,
                            self.img_size,
                        ]
                    ),
                )

    def test_neck(self):
        net = Neck(self.img_size, self.num_classes)
        out1, out2, out3 = net(self.out_52, self.out_26, self.out_13)
        self.assertEqual(
            out1.size(),
            torch.Size(
                [
                    self.batch_size,
                    len(ANCHORS[2]),
                    self.out_13.size(2),
                    self.out_13.size(3),
                    self.num_classes + 5,
                ]
            ),
        )
        self.assertEqual(
            out2.size(),
            torch.Size(
                [
                    self.batch_size,
                    len(ANCHORS[1]),
                    self.out_26.size(2),
                    self.out_26.size(3),
                    self.num_classes + 5,
                ]
            ),
        )
        self.assertEqual(
            out3.size(),
            torch.Size(
                [
                    self.batch_size,
                    len(ANCHORS[0]),
                    self.out_52.size(2),
                    self.out_52.size(3),
                    self.num_classes + 5,
                ]
            ),
        )

    def test_residual_block(self):
        net = ResidualBlock(self.in_channels + 1)
        x = torch.rand(self.batch_size, 4, 4, 4)
        out = net(x)
        self.assertEqual(
            out.size(), torch.Size([self.batch_size, self.in_channels + 1, 4, 4])
        )

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
                out = net(x)
                self.assertEqual(
                    out.size(),
                    torch.Size(
                        [
                            self.batch_size,
                            len(anchors),
                            self.img_size,
                            self.img_size,
                            self.num_classes + 5,
                        ]
                    ),
                )

    def test_yolo_v3(self):
        net = YOLOv3(self.img_size, self.num_classes)
        out1, out2, out3 = net(self.x)
        self.assertEqual(
            out1.size(),
            torch.Size(
                [
                    self.batch_size,
                    len(ANCHORS[2]),
                    self.out_13.size(2),
                    self.out_13.size(3),
                    self.num_classes + 5,
                ]
            ),
        )
        self.assertEqual(
            out2.size(),
            torch.Size(
                [
                    self.batch_size,
                    len(ANCHORS[1]),
                    self.out_26.size(2),
                    self.out_26.size(3),
                    self.num_classes + 5,
                ]
            ),
        )
        self.assertEqual(
            out3.size(),
            torch.Size(
                [
                    self.batch_size,
                    len(ANCHORS[0]),
                    self.out_52.size(2),
                    self.out_52.size(3),
                    self.num_classes + 5,
                ]
            ),
        )


class TestInvalidYOLO(unittest.TestCase):
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
    unittest.main()

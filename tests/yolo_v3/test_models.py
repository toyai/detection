"""Unitteting everything in yolo_v3/models.py."""

from collections import OrderedDict
from unittest import TestCase, main

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import Tensor, nn

from toydet.yolo_v3.models import (
    ANCHORS,
    ConvBN,
    Detector,
    Extractor,
    ResidualBlock,
    YOLOLayer,
    YOLOv3,
    _make_residual_layers,
    six_convbn,
)


def even(number):
    if number % 2 == 0:
        return number


class TestYOLO(TestCase):
    @settings(deadline=None, derandomize=True)
    @given(
        st.integers(1, 3),
        st.integers(3, 32),
        st.integers(1, 3),
        st.integers(12, 608),
    )
    def test_convbn(self, in_c, out_c, kernel, size):
        model = ConvBN(in_c, out_c, kernel)
        result = model(torch.rand(2, in_c, size, size))
        # actually bias is None if pass `False`.
        self.assertEqual(model.conv.bias, None)
        self.assertFalse(torch.isnan(result).all() or torch.isinf(result).all())

    @settings(deadline=None, derandomize=True)
    @given(
        st.integers(32, 128).filter(even),
        st.integers(1, 10),
    )
    def test_make_residual_layers(self, out_c, times):
        result = _make_residual_layers(out_c, times)
        self.assertIsInstance(result, nn.Sequential)
        self.assertEqual(len(result), times)
        for r in result:
            self.assertIsInstance(r, ResidualBlock)

    @settings(deadline=None, derandomize=True)
    @given(
        st.integers(1, 3),
        st.integers(3, 32).filter(even),
        st.tuples(st.integers(), st.integers(), st.integers()),
        st.integers(0, 80),
        st.integers(0, 10),
    )
    def test_six_convbn(self, in_c, out_c, anchor, num_classes, idx):
        result = six_convbn(in_c, out_c, anchor, num_classes, idx)
        self.assertIsInstance(result, OrderedDict)
        self.assertIsInstance(result[f"conv_layer_{idx}"].bias, Tensor)
        self.assertTupleEqual(
            tuple(result.keys()),
            (
                f"module_list_{idx}",
                f"tip_{idx}",
                f"conv_layer_{idx}",
                f"yolo_layer_{idx}",
            ),
        )
        self.assertEqual(len(result), 4)

    @settings(deadline=None, derandomize=True)
    @given(st.sampled_from((416, 608)))
    def test_extractor(self, size):
        model = Extractor()
        result = model(torch.rand(2, 3, size, size))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        for r in result:
            self.assertFalse(torch.isnan(r).all() or torch.isinf(r).all())


class TestInvalidYOLO(TestCase):
    def setUp(self):
        self.batch_size = 1
        self.in_channels = 3
        self.img_size = 416
        self.num_classes = 20
        self.x = torch.rand(
            self.batch_size, self.in_channels, self.img_size, self.img_size + 1
        )

    def test_invalid_six_convbn(self):
        out_channel = 3
        with self.assertRaisesRegex(ValueError, r"out_channels must be divisible by 2"):
            six_convbn(self.in_channels, out_channel, ANCHORS[0], self.num_classes, 0)

    def test_invalid_residual_block(self):
        with self.assertRaisesRegex(ValueError, r"in_channels must be divisible by 2"):
            ResidualBlock(self.in_channels)

    def test_invalid_yolo_layer(self):
        net = YOLOLayer(ANCHORS[0], self.num_classes)
        with self.assertRaisesRegex(ValueError, r"must have same height and width"):
            net(self.x)

    def test_invalid_yolo_v3(self):
        net = YOLOv3(self.img_size, self.num_classes).eval()
        with self.assertRaisesRegex(ValueError, r"must have same height and width"):
            net(self.x)


if __name__ == "__main__":
    main()

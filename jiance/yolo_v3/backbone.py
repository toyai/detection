from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = (
    "ConvBN",
    "ResidualBlock",
    "BackBone",
    "YOLOLayer",
    "MidBlock",
    "Neck",
    "YOLOv3",
)


class ConvBN(nn.Module):
    """
    Convolution2D -> Normalization -> Activation Block.

    Args:
        in_channels (int)
        out_channels (int)
        kernel_size (int)
        norm_layer (Callable): Default BatchNorm2d.
        activation_layer (Callable): Default LeakyReLU
        **conv_kwargs
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        **conv_kwargs: Optional[Dict],
    ) -> None:
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
            padding=(kernel_size - 1) // 2,
            **conv_kwargs,
        )
        self.norm = norm_layer or nn.BatchNorm2d(out_channels)
        self.activation = activation_layer or nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class ResidualBlock(nn.Module):
    """
    Residual 2 ConvBN Blocks.

    Args:
        in_channels (int)
    """

    def __init__(self, in_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        if in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2.")
        half_channels = in_channels // 2
        self.convbn1 = ConvBN(in_channels, half_channels, kernel_size=1)
        self.convbn2 = ConvBN(half_channels, in_channels, kernel_size=3)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        return self.convbn2(self.convbn1(inputs)) + x


def _make_residual_layers(in_channels, out_channels, times):
    """
    Make residual block `times` times.

    Args:
        in_channels (int)
        out_channels (int)
        times (int)

    Returns:
        module_list (nn.Sequential)
    """
    module_list = [ConvBN(in_channels, out_channels, kernel_size=3, stride=2)]
    for _ in range(times):
        module_list.append(ResidualBlock(out_channels))

    return nn.Sequential(*module_list)


class BackBone(nn.Module):
    """
    YOLOv3 BackBone Block.

    Returns:
        output_52, output_26, output_13
    """

    def __init__(self) -> None:
        super(BackBone, self).__init__()
        self.convbn = ConvBN(3, 32, kernel_size=3)
        self.res_block_1x = _make_residual_layers(32, 64, 1)
        self.res_block_2x = _make_residual_layers(64, 128, 2)
        self.res_block_8x_1 = _make_residual_layers(128, 256, 8)
        self.res_block_8x_2 = _make_residual_layers(256, 512, 8)
        self.res_block_4x = _make_residual_layers(512, 1024, 4)

    def forward(self, inputs: Tensor) -> Tuple[Tensor]:
        x = self.res_block_2x(self.res_block_1x(self.convbn(inputs)))
        output_52 = self.res_block_8x_1(x)
        output_26 = self.res_block_8x_2(output_52)
        output_13 = self.res_block_4x(output_26)

        return output_52, output_26, output_13


class YOLOLayer(nn.Module):
    """
    YOLOv3 YOLO Layer.

    Args:
        anchors (Sequence[int])
        img_size (int)
        num_classes (int)

    Returns:
        prediction (Tensor) [B, A, H, W, C + 5]
    """

    def __init__(self, anchors: Sequence[int], img_size: int, num_classes: int) -> None:
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.img_size = img_size
        self.num_anchors = len(anchors)
        self.num_classes = num_classes

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.size(2) != inputs.size(3):
            raise ValueError("image must have same height and width.")

        batch_size = inputs.size(0)
        grid_size = inputs.size(2)  # 13x13, 26x26, 52x52

        # B - batch_size, A - num_anchors, C - num_classes, H - height, W - width
        # (B, A, C + 5, H, W) -> (B, A, H, W, C + 5)
        # cx, cy are relative values meaning they are between 0 and 1
        # w, h can be greater than 0
        pred = inputs.reshape(
            batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size
        ).permute(0, 1, 3, 4, 2)

        # still relative values
        # cxcy = torch.sigmoid(pred[..., :2])
        # wh = pred[..., 2:4]
        # pred_conf = pred[..., 4]
        # pred_cls = pred[..., 5:]

        return pred


class MidBlock(nn.Module):
    """
    Middle block of YOLOve Neck.

    Args:
        in_channels (int)
        out_channels (int)
        anchors (Sequence[int])
        img_size (int)
        num_classes (int)

    Returns:
        inputs, branch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        anchors: Sequence[int],
        img_size: int,
        num_classes: int,
    ) -> None:
        super(MidBlock, self).__init__()
        if out_channels % 2 != 0:
            raise ValueError("out_channels must be divisible by 2.")
        half_channels = out_channels // 2
        self.convbn1 = ConvBN(in_channels, half_channels, kernel_size=1)
        self.convbn2 = ConvBN(half_channels, out_channels, kernel_size=3)
        self.convbn3 = ConvBN(out_channels, half_channels, kernel_size=1)
        self.convbn4 = ConvBN(half_channels, out_channels, kernel_size=3)
        self.convbn5 = ConvBN(out_channels, half_channels, kernel_size=1)
        self.convbn6 = ConvBN(half_channels, out_channels, kernel_size=3)
        self.conv = nn.Conv2d(
            out_channels, (num_classes + 5) * len(anchors), 1, bias=True
        )
        self.yolo_layer = YOLOLayer(anchors, img_size, num_classes)

    def forward(self, inputs: Tensor) -> Tuple[Tensor]:
        branch = self.convbn5(
            self.convbn4(self.convbn3(self.convbn2(self.convbn1(inputs))))
        )
        inputs = self.yolo_layer(self.conv(self.convbn6(branch)))

        return inputs, branch


class Neck(nn.Module):
    """
    YOLOv3 Neck.

    Args:
        img_size (int)
        num_classes (int)

    Returns:
        out1, out2, out3
    """

    def __init__(self, img_size: int, num_classes: int) -> None:
        super(Neck, self).__init__()
        ANCHORS = (
            ((10, 13), (16, 30), (33, 23)),
            ((30, 61), (62, 45), (59, 119)),
            ((116, 90), (156, 198), (373, 326)),
        )
        self.block1 = MidBlock(1024, 1024, ANCHORS[-1], img_size, num_classes)
        self.convbn1 = ConvBN(512, 256, kernel_size=1)
        self.block2 = MidBlock(768, 512, ANCHORS[1], img_size, num_classes)
        self.convbn2 = ConvBN(256, 128, kernel_size=1)
        self.block3 = MidBlock(384, 256, ANCHORS[0], img_size, num_classes)

    def forward(
        self, input_52: Tensor, input_26: Tensor, input_13: Tensor
    ) -> Tuple[Tensor]:
        out1, branch = self.block1(input_13)
        x = self.convbn1(branch)
        x = F.interpolate(x, scale_factor=2.0)
        x = torch.cat((input_26, x), dim=1)
        out2, branch = self.block2(x)
        x = self.convbn2(branch)
        x = F.interpolate(x, scale_factor=2.0)
        x = torch.cat((input_52, x), dim=1)
        out3, _ = self.block3(x)

        return out1, out2, out3


class YOLOv3(nn.Module):
    """
    The Final YOLOv3 Model.

    Args:
        img_size (int)
        num_classes (int)

    Returns:
        out1, out2, out3
    """

    def __init__(self, img_size: int, num_classes: int) -> None:
        super(YOLOv3, self).__init__()
        self.backbone = BackBone()
        self.neck = Neck(img_size, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        output_52, output_26, output_13 = self.backbone(inputs)
        out1, out2, out3 = self.neck(output_52, output_26, output_13)

        return out1, out2, out3


ANCHORS = (
    ((10, 13), (16, 30), (33, 23)),
    ((30, 61), (62, 45), (59, 119)),
    ((116, 90), (156, 198), (373, 326)),
)
# net = YOLOv3(ANCHORS[0], 416, 20)
x = torch.rand(1, 3, 416, 416)

# a, b, c = net(x)

# print(a.shape, b.shape, c.shape)

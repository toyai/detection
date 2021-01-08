"""YOLO v3 Models."""

import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from ignite.distributed import device
from ignite.utils import to_onehot
from torch import Tensor, nn
from torchvision.ops import box_convert

from toydet.yolo_v3.utils import get_abs_yolo_bbox, box_iou_wh

__all__ = (
    "ANCHORS",
    "ConvBN",
    "ResidualBlock",
    "YOLOLayer",
    "YOLOv3",
)

logger = logging.getLogger(__name__)


ANCHORS = (
    ((10, 13), (16, 30), (33, 23)),
    ((30, 61), (62, 45), (59, 119)),
    ((116, 90), (156, 198), (373, 326)),
)


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
    module_list = []
    for _ in range(times):
        module_list.append(ResidualBlock(out_channels))

    return nn.Sequential(*module_list)


def six_convbn(in_channels, out_channels, anchors, num_classes, idx):
    if out_channels % 2 != 0:
        raise ValueError("out_channels must be divisible by 2.")
    half_channels = out_channels // 2
    module_dict = OrderedDict()
    module_dict[f"module_list_{idx}"] = nn.Sequential(
        ConvBN(in_channels, half_channels, kernel_size=1),
        ConvBN(half_channels, out_channels, kernel_size=3),
        ConvBN(out_channels, half_channels, kernel_size=1),
        ConvBN(half_channels, out_channels, kernel_size=3),
        ConvBN(out_channels, half_channels, kernel_size=1),
    )
    module_dict[f"tip_{idx}"] = ConvBN(half_channels, out_channels, kernel_size=3)
    module_dict[f"conv_layer_{idx}"] = nn.Conv2d(
        out_channels, (num_classes + 5) * len(anchors), kernel_size=1, bias=True
    )
    module_dict[f"yolo_layer_{idx}"] = YOLOLayer(anchors, num_classes)
    return module_dict


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
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
            padding=(kernel_size - 1) // 2,
            **conv_kwargs,
        )
        self.norm = norm_layer or nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        self.activation = activation_layer or nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        ConvBN forward function.

        Args:
            inputs (Tensor)
        """
        return self.activation(self.norm(self.conv(inputs)))


class ResidualBlock(nn.Module):
    """
    Residual 2 ConvBN Blocks.

    Args:
        in_channels (int)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError("in_channels must be divisible by 2.")
        half_channels = in_channels // 2
        self.convbn1 = ConvBN(in_channels, half_channels, kernel_size=1)
        self.convbn2 = ConvBN(half_channels, in_channels, kernel_size=3)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        ResidualBlock forward function.

        Args:
            inputs (Tensor)
        """
        shortcut = inputs
        return self.convbn2(self.convbn1(inputs)) + shortcut


class Extractor(nn.Module):
    """
    YOLOv3 BackBone Block.

    Returns:
        list: results (52, 26, 13)
    """

    def __init__(self) -> None:
        super().__init__()
        self.module_dict = nn.ModuleDict({"conv_block_0": ConvBN(3, 32, kernel_size=3)})
        bbs = (32, 64, 128, 256, 512, 1024)
        times = (1, 2, 8, 8, 4)
        for i, (bb, time) in enumerate(zip(bbs, times)):
            self.module_dict[f"conv_block_{i + 1}"] = ConvBN(
                bb, bbs[i + 1], kernel_size=3, stride=2
            )
            self.module_dict[f"res_block_{i + 1}"] = _make_residual_layers(
                bb, bbs[i + 1], time
            )

    def forward(self, inputs: Tensor) -> List[Tensor]:
        """
        YOLOv3 BackBone forward function.

        Args:
            inputs (Tensor)

        Returns:
            list: results (52, 26, 13)
        """
        results = []
        for name, module in self.module_dict.items():
            inputs = module(inputs)
            if name in ("res_block_3", "res_block_4", "res_block_5"):
                results.append(inputs)

        return results


class YOLOLayer(nn.Module):
    """
    YOLOv3 YOLO Layer.

    Args:
        anchors (Tensor)
        img_size (int)
        num_classes (int)

    Returns:
        prediction (Tensor) [B, A, H, W, C + 5]
    """

    def __init__(self, anchors: Tensor, num_classes: int) -> None:
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.grid_size = 0
        self.bce_loss_with_logits = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs: Tensor, target: Tensor = None) -> Tensor:
        """
        YOLOLayer forward function.

        Args:
            inputs (Tensor)

        Returns:
            Tensor: pred_bbox, pred_cls, pred_conf in train mode
            Tensor: pred_bbox * stride, sigmoid(pred_cls), sigmoid(pred_conf) if eval
        """
        if inputs.size(2) != inputs.size(3):
            raise ValueError("image must have same height and width.")

        batch_size = inputs.size(0)
        grid_size = inputs.size(2)  # 13x13, 26x26, 52x52 for 416 img size

        # B - batch_size, A - num_anchors, C - num_classes, H - height, W - width
        # [B, A, H, W, C + 5]
        # inputs: [B, A * (C + 5), H, W]
        # pred: [B, A, H, W, C + 5]
        pred = (
            inputs.reshape(
                batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # cx, cy - center x, center y
        # cx, cy are relative values meaning they are between 0 and 1
        # w, h can be greater than 1
        # make into relative values, sigmoid(tx,ty) from paper
        pred[..., :2] = torch.sigmoid(pred[..., :2])

        if self.grid_size != grid_size:
            self.grid_size = grid_size
            # anchors are already divided by image size, so its [0-1]
            # multiply by grid_size for grid_size x grid_size output
            self.anchors = self.anchors * self.grid_size
            logger.info(
                "Multiplied normalized anchors with grid size %i ...", grid_size
            )

        if self.training:
            # target: [number of objects in a batch, 6]
            # target is also normalized by img_size, so its [0-1]
            # multiplied with grid_size for grid_size x grid_size output
            target = torch.cat(
                (target[..., :2], target[..., 2:] * self.grid_size), dim=-1
            )
            return self.build_targets(pred, target)
            # loss_xywh = self.mse_loss(pred["bbox"], target["bbox"])
            # loss_cls = self.bce_loss(pred["cls"], target["cls"])
            # loss_obj = self.bce_loss_with_logits(pred["obj"], target["obj"])
            # loss_noobj = self.bce_loss_with_logits(pred["noobj"], target["noobj"])
            # loss_conf = loss_obj + loss_noobj
            # losses = loss_cls + loss_conf + loss_xywh
            # return {
            #     "loss/xywh": loss_xywh.detach().cpu().item(),
            #     "loss/cls": loss_cls.detach().cpu().item(),
            #     "loss/conf": loss_conf.detach().cpu().item(),
            #     "loss/total": losses,
            # }

        rel_bbox, rel_conf, rel_cls = torch.split(
            pred, (4, 1, self.num_classes), dim=-1
        )
        # divide by grid_size so they are in [0-1]
        # and later we can multiply with img_size for img_size x img_size output
        abs_bbox = get_abs_yolo_bbox(rel_bbox, self.anchors) / self.grid_size
        abs_conf = torch.sigmoid(rel_conf)
        abs_cls = torch.sigmoid(rel_cls)

        return (
            torch.cat(
                (
                    abs_bbox.reshape(batch_size, -1, 4),
                    abs_conf.reshape(batch_size, -1, 1),
                    abs_cls.reshape(batch_size, -1, self.num_classes),
                ),
                dim=-1,
            ),
            0,
        )

    def build_targets(
        self,
        pred: Tensor,
        target: Tensor,
        iou_ignore_threshold: float = 0.5,
    ):
        target_xy = torch.narrow(target, dim=-1, start=2, length=2)
        target_wh = torch.narrow(target, dim=-1, start=4, length=2)
        batch, labels = torch.narrow(target, dim=-1, start=0, length=2).long().t()
        width, height = target_xy.long().t()

        ious = torch.stack(
            [box_iou_wh(anchor, target_wh) for anchor in self.anchors], dim=0
        )
        _, iou_idx = torch.max(ious, 0)

        target_xy = target_xy - torch.floor(target_xy)
        target_wh = torch.log(target[..., 4:] / self.anchors[iou_idx])
        pred_bbox = pred[batch, iou_idx, height, width, :4]
        target_bbox = torch.cat((target_xy, target_wh), dim=-1)
        if not pred_bbox.shape == target_bbox.shape:
            raise AssertionError(
                f"Got pred_bbox {pred_bbox.shape}, target_bbox {target_bbox.shape}"
            )
        target_conf = torch.ones((target.shape[0], 1), device=target.device)
        target_cls = to_onehot(labels.to(torch.int64), self.num_classes)
        target = torch.cat((target_bbox, target_conf, target_cls), dim=-1)
        return pred[batch, iou_idx, height, width, :], target
        # target_bbox = torch.zeros_like(pred_bbox)
        # target_cls = torch.zeros_like(pred_cls)
        # # ious_scores = torch.zeros_like(pred_cls, device=pred_cls.device)
        # obj_mask = torch.zeros_like(pred_conf, dtype=torch.bool)
        # noobj_mask = torch.ones_like(pred_conf, dtype=torch.bool)

        # # obj_mask = 1 => there is obj
        # # obj_mask = 0 => there is no obj
        # # noobj_mask = 1 => there is no obj
        # # noobj_mask = 0 => there is obj
        # noobj_mask[batch, idx, height, width, :] = 0
        # for i, iou in enumerate(ious.t()):
        #     noobj_mask[batch[i], iou > iou_ignore_threshold, height[i], width[i], :] = 0

        # # target_bbox[batch, idx, height, width, :2] = cxcy_t - torch.floor(cxcy_t)
        # # target_bbox[batch, idx, height, width, 2:] = torch.log(wh_t / anchors[idx] + 1e-16)
        # # ious_scores[batch, idx, :, height, width] = box_iou(pred_bbox, target_bbox)

        # target_conf = obj_mask.float()
        # # pred_bbox = torch.masked_select(pred_bbox, obj_mask)
        # pred_cls = torch.masked_select(pred_cls, obj_mask)
        # pred_obj = torch.masked_select(pred_conf, obj_mask)
        # pred_noobj = torch.masked_select(pred_conf, noobj_mask)
        # # target_bbox = torch.masked_select(target_bbox, obj_mask)
        # target_cls = torch.masked_select(target_cls, obj_mask)
        # target_obj = torch.masked_select(target_conf, obj_mask)
        # target_noobj = torch.masked_select(target_conf, noobj_mask)

        # pred = {
        #     "bbox": pred_bbox[batch, idx, height, width, :],
        #     "cls": pred_cls,
        #     "obj": pred_obj,
        #     "noobj": pred_noobj,
        # }
        # target = {
        #     "bbox": target_bbox,
        #     "cls": target_cls,
        #     "obj": target_obj,
        #     "noobj": target_noobj,
        # }
        # return pred, target


class Detector(nn.Module):
    """
    YOLOv3 Detector.

    Args:
        img_size (int)
        num_classes (int)

    Returns:
        out1, out2, out3 (Tuple[Tensor])
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # pylint: disable=not-callable
        anchors = (
            torch.tensor(
                [
                    ((116, 90), (156, 198), (373, 326)),
                    ((30, 61), (62, 45), (59, 119)),
                    ((10, 13), (16, 30), (33, 23)),
                ],
                device=device(),
            )
            / 416.0
        )
        self.module_dict = nn.ModuleDict()
        channels = ((1024, 1024), (768, 512), (384, 256))
        for idx, (channel, anchor) in enumerate(zip(channels, anchors)):
            self.module_dict.update(
                six_convbn(
                    *channel,
                    anchors=anchor,
                    num_classes=num_classes,
                    idx=idx,
                )
            )
            if idx != 2:
                in_channels = channels[idx + 1][-1]
                out_channels = in_channels // 2
                self.module_dict[f"conv_block_{idx}"] = ConvBN(
                    in_channels, out_channels, kernel_size=1
                )
                self.module_dict[f"upsample_{idx}"] = nn.Upsample(scale_factor=2.0)

    def forward(self, inputs: Tensor, target: Tensor = None) -> Tuple[Tensor]:
        """
        YOLOv3 Neck Block forward function.

        Args:
            input_52 (Tensor)
            input_26 (Tensor)
            input_13 (Tensor)

        Returns:
            Tuple[Tensor]: out1, out2, out3
        """
        pred_list, target_list = [], []
        output = inputs[-1]
        idx = 1
        for name, module in self.module_dict.items():
            if "module_list" in name:
                output = module(output)
                tip = output
            elif "yolo_layer" in name:
                yolo_layer_output = module(output, target)
                pred_list.append(yolo_layer_output[0])
                target_list.append(yolo_layer_output[1])
                # results.append(module(output, target))
            elif "conv_block" in name:
                output = module(tip)
            elif "upsample" in name:
                output = module(output)
                output = torch.cat((output, inputs[idx]), dim=1)
                idx -= 1
            else:
                output = module(output)
        del tip
        return pred_list, target_list


class YOLOv3(nn.Module):
    """
    The Final YOLOv3 Model.

    Args:
        img_size (int)
        num_classes (int)

    Returns:
        out1, out2, out3 (Tuple[Tensor])
    """

    def __init__(self, img_size: int, num_classes: int) -> None:
        super().__init__()
        self.img_size = img_size
        self.extractor = Extractor()
        self.detector = Detector(num_classes)
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs: Tensor, target: Tensor = None) -> Tensor:
        """
        YOLOv3 forward function.

        Args:
            inputs (Tensor)

        Returns:
            Tuple[Tensor]: out1, out2, out3
        """
        results = self.extractor(inputs)
        if self.training:
            target = torch.cat(
                (
                    target[:, :2],
                    box_convert(target[:, 2:], "xyxy", "cxcywh") / self.img_size,
                ),
                dim=-1,
            )
        pred_list, target_list = self.detector(results, target)
        # return results
        if self.training:
            pred_list = torch.cat(pred_list, dim=0)
            target_list = torch.cat(target_list, dim=0)
            loss_xywh = self.mse_loss(pred_list[..., :4], target_list[..., :4])
            loss_conf = self.bce_with_logits_loss(
                pred_list[..., 4], target_list[..., 4]
            )
            loss_cls = self.bce_with_logits_loss(
                pred_list[..., 5:], target_list[..., 5:]
            )
            total_loss = loss_xywh + loss_cls + loss_conf
            loss_dict = {
                "loss/xywh": loss_xywh.detach().cpu().item(),
                "loss/cls": loss_cls.detach().cpu().item(),
                "loss/conf": loss_conf.detach().cpu().item(),
                "loss/total": total_loss.detach().cpu().item(),
            }
            return loss_dict, total_loss

        return pred_list

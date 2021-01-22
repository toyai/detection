"""YOLO v3 Models."""

from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from toydet.yolo_v3.utils import box_iou_wh, parse_config


def create_modules(configs: List[dict]):
    module_list = nn.ModuleList()
    hyp = configs.pop(0)
    output_filters = [int(hyp["channels"])]
    for i, config in enumerate(configs):
        module_seq = nn.Sequential()
        if config["type"] == "convolutional":
            bn = int(config["batch_normalize"])
            filters = int(config["filters"])
            kernel_size = int(config["size"])
            padding = (kernel_size - 1) // 2
            module_seq.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=int(config["stride"]),
                    bias=not bn,
                ),
            )
            if bn:
                module_seq.add_module(
                    f"bn_{i}", nn.BatchNorm2d(filters, 1e-5, float(hyp["momentum"]))
                )
            if config["activation"] == "leaky":
                module_seq.add_module(f"leaky_{i}", nn.LeakyReLU(0.1, True))

        elif config["type"] == "maxpool":
            kernel_size = int(config["size"])
            stride = int(config["stride"])
            padding = (kernel_size - 1) // 2
            if kernel_size == 2 and stride == 1:
                module_seq.add_module(f"_debug_padding_{i}", nn.ZeroPad2d((0, 1, 0, 1)))
            module_seq.add_module(
                f"max_pool_{i}",
                nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            )

        elif config["type"] == "upsample":
            module_seq.add_module(
                f"upsample_{i}", nn.Upsample(scale_factor=int(config["stride"]))
            )

        elif config["type"] == "route":
            layers = [int(x) for x in config["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            module_seq.add_module(f"route_{i}", nn.Sequential())

        elif config["type"] == "shortcut":
            filters = output_filters[1:][int(config["from"])]
            module_seq.add_module(f"shortcut_{i}", nn.Sequential())

        elif config["type"] == "yolo":
            anchor_idxs = [int(i) for i in config["mask"].split(",")]
            anchors = [int(i) for i in config["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(config["classes"])
            ignore_thres = float(config["ignore_thresh"])
            img_size = int(hyp["height"])
            module_seq.add_module(
                f"yolo_{i}", YOLOLayer(anchors, num_classes, img_size, ignore_thres)
            )

        output_filters.append(filters)
        module_list.append(module_seq)

    return hyp, module_list


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size, ignore_thres):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.ignore_thres = ignore_thres
        self.num_classes = num_classes
        self.grid_size = 0
        self.stride = 0.0
        self.img_size = img_size

    def forward(self, inputs: Tensor, target: Tensor = None):
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
            self.stride = self.img_size / grid_size
            aranged_tensor = torch.arange(grid_size, device=pred.device)
            self.grid_xy = torch.stack(
                torch.meshgrid(aranged_tensor, aranged_tensor)[::-1], dim=-1
            ).reshape(1, 1, grid_size, grid_size, 2)
            self.anchors = torch.tensor(
                [(w / self.stride, h / self.stride) for w, h in self.anchors],
                device=pred.device,
            )
            del aranged_tensor

        x_y, w_h, conf, cls_ = torch.split(pred, (2, 2, 1, self.num_classes), dim=-1)
        x_y = torch.add(x_y, self.grid_xy) * self.stride
        w_h = torch.exp(w_h) * self.anchors.reshape(1, 3, 1, 1, 2) * self.stride
        output = torch.cat(
            (
                x_y.reshape(batch_size, -1, 2),
                w_h.reshape(batch_size, -1, 2),
                torch.sigmoid(conf).reshape(batch_size, -1, 1),
                torch.sigmoid(cls_).reshape(batch_size, -1, self.num_classes),
            ),
            dim=-1,
        )
        if self.training:
            total_loss, loss_tuple = self.train_loss(pred, target)
            return output, total_loss, loss_tuple
        return output, 0, 0

    def train_loss(
        self,
        pred: Tensor,
        target: Tensor,
    ):
        target = torch.cat((target[..., :2], target[..., 2:] * pred.size(2)), dim=-1)
        pred_bbox, pred_conf, pred_cls = torch.split(
            pred, (4, 1, self.num_classes), dim=-1
        )
        batch, labels, target_xy, target_wh = torch.split(target, (1, 1, 2, 2), dim=-1)
        batch, labels = batch.long().squeeze(-1), labels.long().squeeze(-1)
        width, height = target_wh[..., 0].long(), target_wh[..., 1].long()

        ious = torch.stack(
            [box_iou_wh(anchor, target_wh) for anchor in self.anchors], dim=0
        )
        _, iou_idx = torch.max(ious, 0)

        target_xy = target_xy - torch.floor(target_xy)
        target_wh = torch.log(target_wh / (self.anchors[iou_idx] + 1e-16))
        pred_bbox = pred_bbox[batch, iou_idx, height, width, :]
        target_bbox = torch.cat((target_xy, target_wh), dim=-1)
        if not pred_bbox.shape == target_bbox.shape:
            raise AssertionError(
                f"Got pred_bbox {pred_bbox.shape}, target_bbox {target_bbox.shape}"
            )
        target_cls = F.one_hot(labels, self.num_classes).type_as(pred_cls)

        obj_mask = torch.zeros_like(pred_conf, dtype=torch.bool)
        obj_mask[batch, iou_idx, height, width, :] = 1
        noobj_mask = 1 - obj_mask.float()
        for i, iou in enumerate(ious.t()):
            noobj_mask[batch[i], iou > self.ignore_thres, height[i], width[i], :] = 0

        pred_obj = torch.masked_select(pred_conf, obj_mask)
        pred_noobj = torch.masked_select(pred_conf, noobj_mask.to(torch.bool))
        target_obj = torch.ones_like(pred_obj)
        target_noobj = torch.zeros_like(pred_noobj)
        pred_conf = torch.cat((pred_obj, pred_noobj), dim=0)
        target_conf = torch.cat((target_obj, target_noobj), dim=0)
        loss_xywh = self.mse_loss(pred_bbox, target_bbox)
        loss_conf = self.bce_with_logits_loss(pred_conf, target_conf)
        loss_cls = self.bce_with_logits_loss(
            pred_cls[batch, iou_idx, height, width, :], target_cls
        )
        losses = loss_xywh + loss_conf + loss_cls
        return losses, (
            loss_xywh.detach().cpu().item(),
            loss_conf.detach().cpu().item(),
            loss_cls.detach().cpu().item(),
            losses.detach().cpu().item(),
        )


class YOLOv3(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.configs = parse_config(name)
        self.hyp, self.module_list = create_modules(self.configs)

    def forward(self, input: Tensor, target: Tensor = None):
        loss = 0
        loss_dict = {"loss/bbox": 0, "loss/conf": 0, "loss/cls": 0, "loss/total": 0}
        layer_outputs, yolo_outputs = [], []
        for config, module in zip(self.configs, self.module_list):
            if config["type"] in ("convolutional", "upsample", "maxpool"):
                input = module(input)
            elif config["type"] == "route":
                input = torch.cat(
                    [layer_outputs[int(i)] for i in config["layers"].split(",")],
                    dim=1,
                )
            elif config["type"] == "shortcut":
                input = layer_outputs[-1] + layer_outputs[int(config["from"])]
            elif config["type"] == "yolo":
                input, layer_loss, loss_tuple = module[0](input, target)
                yolo_outputs.append(input)
                if self.training:
                    loss += layer_loss
                    loss_dict["loss/bbox"] += loss_tuple[0]
                    loss_dict["loss/conf"] += loss_tuple[1]
                    loss_dict["loss/cls"] += loss_tuple[2]
                    loss_dict["loss/total"] += loss_tuple[3]
            layer_outputs.append(input)
        yolo_outputs = torch.cat(yolo_outputs, dim=1)
        if self.training:
            return loss, yolo_outputs, loss_dict
        return yolo_outputs

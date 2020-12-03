"""
YOLO v3 Model
"""

import torch
from torch import nn
from torchvision.ops import box_convert

from jiance.yolo_v3.utils import parse_cfg, wh_bbox_iou


def create_yolo_v3(modules):
    net_cfg = modules.pop(0)
    output_filters = [int(net_cfg["channels"])]
    module_list = nn.ModuleList()

    for i, module in enumerate(modules):
        module_seq = nn.Sequential()

        if module["type"] == "convolutional":
            bn = int(module["batch_normalize"])
            filters = int(module["filters"])
            kernel_size = int(module["size"])
            stride = int(module["stride"])
            padding = (kernel_size - 1) // 2
            module_seq.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    output_filters[-1],
                    filters,
                    kernel_size,
                    stride,
                    padding,
                    bias=not bn,
                ),
            )
            if bn:
                module_seq.add_module(f"batch_norm_{i}", nn.BatchNorm2d(filters, 0.9))
            if module["activation"] == "leaky":
                module_seq.add_module(f"leaky_{i}", nn.LeakyReLU(0.1, True))

        elif module["type"] == "maxpool":
            kernel_size = int(module["size"])
            stride = int(module["stride"])
            padding = (kernel_size - 1) // 2
            module_seq.add_module(
                f"max_pool_{i}", nn.MaxPool2d(kernel_size, stride, padding)
            )

        elif module["type"] == "upsample":
            stride = int(module["stride"])
            module_seq.add_module(f"upsample_{i}", nn.Upsample(scale_factor=stride))

        elif module["type"] == "shortcut":
            filters = output_filters[1:][int(module["from"])]
            module_seq.add_module(f"shortcut_{i}", EmptyLayer())

        elif module["type"] == "route":
            layers = [int(x) for x in module["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            module_seq.add_module(f"route_{i}", EmptyLayer())

        elif module["type"] == "yolo":
            mask = [int(m) for m in module["mask"].split(",")]
            anchors = [int(a) for a in module["anchors"].split(",")]
            anchors = [(anchors[a], anchors[a + 1]) for a in range(0, len(anchors), 2)]
            anchors = [anchors[m] for m in mask]
            num_classes = int(module["classes"])
            img_size = int(net_cfg["width"])

            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            module_seq.add_module(f"yolo_{i}", yolo_layer)

        module_list.append(module_seq)
        output_filters.append(filters)

    return net_cfg, module_list


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = 0
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x, targets=None):
        if x.size(2) != x.size(3):
            raise ValueError("image must have same height and width.")

        batch_size = x.size(0)
        grid_size = x.size(2)  # 13x13, 26x26, 52x52

        # B - batch_size, A - num_anchors, C - num_classes, H - height, W - width
        # (B, A, C + 5, H, W) -> (B, A, H, W, C + 5)
        # cx, cy are relative values meaning they are between 0 and 1
        # w, h can be greater than 0
        pred = x.reshape(
            batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size
        ).permute(0, 1, 3, 4, 2)

        # still relative values
        cxcy = torch.sigmoid(pred[..., :2])
        wh = pred[..., 2:4]
        pred_conf = pred[..., 4]
        pred_cls = pred[..., 5:]

        if self.grid_size != grid_size:
            print(f"Recomputing for grid size {grid_size}...")
            self.get_grid_offsets(grid_size)

        # get absolute values with equation from the paper
        abs_cxcy = torch.add(cxcy, self.grid_xy)
        abs_wh = torch.exp(wh) * self.grid_wh
        pred_bbox = torch.cat((abs_cxcy, abs_wh), dim=-1)

        if not self.training:
            return (
                pred_bbox * self.stride,  # to get the actual cxcywh on img_size image
                torch.sigmoid(pred_conf),
                torch.sigmoid(pred_cls),
            )

        cxcy_t, wh_t, cls_t, conf_t, obj_mask = self.build_cxcywh_target(
            cxcy, wh, pred_cls, pred_conf, targets
        )
        loss_xy = self.mse_loss(cxcy, cxcy_t)
        loss_wh = self.mse_loss(wh, wh_t)
        loss_conf = self.bce_with_logits_loss(pred_conf, conf_t)
        loss_cls = self.bce_with_logits_loss(pred_cls, cls_t)
        losses = loss_xy + loss_wh + loss_conf + loss_cls
        metrics = None

        return (pred_bbox, pred_conf, pred_cls), losses, metrics

    def get_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        self.stride = self.img_size / self.grid_size
        t = torch.arange(grid_size, device=self.device)
        y, x = torch.meshgrid(t, t)
        # (1, 1, 13, 13, 2)
        self.grid_xy = torch.stack((x, y), dim=-1).unsqueeze(0).unsqueeze(0)
        self.scaled_anchors = torch.tensor(
            [(w / self.stride, h / self.stride) for w, h in self.anchors],
            device=self.device,
        )
        # (1, 3, 1, 1, 2)
        self.grid_wh = self.scaled_anchors.reshape(1, self.num_anchors, 1, 1, 2)

    def build_cxcywh_target(self, cxcy, wh, pred_cls, pred_conf, targets):
        # cxcywh - N x A x H x W x 2
        # wh - N x A x H x W x 2
        # pred_cls - N x A x H x W x 20
        # pred_conf - N x A x H x W x 1
        # targets - Tuple[Sequence[List[Sequence[List]]]]
        # targets - ([[], []], [[], [], []])
        # anchors - 3 x 2

        # cxcy_t = torch.rand_like(cxcy)
        # wh_t = torch.rand_like(wh)
        # conf_t = torch.rand_like(pred_conf)
        # cls_t = torch.rand_like(pred_cls)

        cxcy_t = torch.zeros_like(cxcy, device=self.device)
        wh_t = torch.zeros_like(wh, device=self.device)
        conf_t = torch.zeros_like(pred_conf, device=self.device)
        cls_t = torch.zeros_like(pred_cls, device=self.device)
        obj_mask = torch.zeros(cxcy[..., 0].shape).long()

        for batch_idx, target in enumerate(targets):
            target = torch.tensor(target, dtype=cxcy.dtype, device=self.device)
            cxcywh, class_idx = target[:, :-1], target[:, -1]
            cxcywh = torch.div(box_convert(cxcywh, "xyxy", "cxcywh"), self.stride)

            cxcy_ = cxcywh[:, :2]
            wh = cxcywh[:, 2:]
            ious = torch.stack(
                [wh_bbox_iou(anchor, wh) for anchor in self.scaled_anchors]
            )
            best_ious, best_n = ious.max(0)
            H, W = cxcy_.long().t()

            # obj_mask[batch_idx, best_n, H, W] = 1
            cxcy_t[batch_idx, best_n, H, W, :] = cxcy_ - cxcy_.floor()
            wh_t[batch_idx, best_n, H, W, :] = torch.log(
                wh / self.scaled_anchors[best_n] + 1e-16
            )
            conf_t[batch_idx, best_n, H, W] = 1
            cls_t[batch_idx, best_n, H, W, class_idx.long()] = 1

        # cxcywh = cxcywh * (cxcy.size(2) / 416)  # grid_size cxcywh
        # cxcy_ = cxcywh[..., :2].squeeze(0)
        # wh = cxcywh[..., 2:]
        # batch_size, class_idx = target[0, ...].long(), target[..., -1]
        # h, w = cxcy_.long().t()

        # cxcy_t[batch_size, 1, h, w, 0] = cxcy_[0] - cxcy_[0].floor()
        # cxcy_t[batch_size, 1, h, w, 1] = cxcy_[1] - cxcy_[1].floor()
        # wh_t[batch_size, 1, h, w, :] = torch.log(wh / anchors + 1e-16)
        # conf_t[..., :] = 1 if class_idx > 0 else 0
        # cls_t[..., :] = class_idx

        return cxcy_t, wh_t, cls_t, conf_t, obj_mask


class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()


class YOLOv3(nn.Module):
    def __init__(self, cfg: str):
        super().__init__()
        self.modules_def = parse_cfg(cfg)
        self.net_cfg, self.module_list = create_yolo_v3(self.modules_def)

    def forward(self, x, targets=None):
        loss = 0
        layer_outputs, yolo_outputs = [], {}
        for i, (module_def, module) in enumerate(
            zip(self.modules_def, self.module_list)
        ):
            if module_def["type"] in ("convolutional", "upsample", "maxpool"):
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat(
                    [layer_outputs[int(i)] for i in module_def["layers"].split(",")],
                    dim=1,
                )
            elif module_def["type"] == "shortcut":
                x = layer_outputs[-1] + layer_outputs[int(module_def["from"])]
            elif module_def["type"] == "yolo":
                x, losses, _ = module[0](x, targets)
                loss += losses
                # yolo_outputs.append(x)
                yolo_outputs[i] = x
            layer_outputs.append(x)

        return yolo_outputs, loss

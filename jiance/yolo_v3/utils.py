"""Common utilites to YOLO v3."""
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def parse_cfg(cfg: str):
    with open(cfg, "r") as f:
        lines = f.read().split("\n")
        lines = [line.strip() for line in lines if line and not line.startswith("#")]

    modules = []
    for line in lines:
        if line.startswith("["):
            modules.append({})
            modules[-1]["type"] = line.strip("[]")
            if modules[-1]["type"] == "convolutional":
                modules[-1]["batch_normalize"] = 0
        else:
            k, v = line.split("=")
            modules[-1][k.strip()] = v.strip()

    return modules


def wh_bbox_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def yolo_loss(pred: Tuple, targets: list, img_size: int, scaled_anchors: Tensor):
    pred_bbox, pred_cls, pred_conf = pred
    target_bbox = torch.zeros_like(pred_bbox, device=pred_bbox.device)
    target_cls = torch.zeros_like(pred_cls, device=pred_cls.device)
    target_conf = torch.zeros_like(pred_conf, device=pred_conf.device)

    target_cxcy, target_wh = torch.split(target_bbox, (2, 2), dim=-1)
    pred_cxcy, pred_wh = torch.split(pred_bbox, (2, 2), dim=-1)

    for batch_idx, target in enumerate(targets):
        target = torch.tensor(target, dtype=pred_cxcy.dtype, device=pred_cxcy.device)
        # target needs to be * grid_size / img_size
        target = target * pred_bbox.size(2) / img_size
        cxcy_t, wh_t, cls_t = torch.split(target, (2, 2, 1), dim=-1)
        ious = torch.stack([wh_bbox_iou(anchor, wh_t) for anchor in scaled_anchors])
        best_ious, best_n = ious.max(0)
        height, width = cxcy_t.long().t()

        target_cxcy[batch_idx, best_n, height, width, :] = cxcy_t - torch.floor(cxcy_t)
        target_wh[batch_idx, best_n, height, width, :] = torch.log(
            wh_t / scaled_anchors[best_n] + 1e-16
        )
        target_conf[batch_idx, best_n, height, width, :] = 1
        target_cls[batch_idx, best_n, height, width, cls_t.long()] = 1

    loss_xy = F.mse_loss(pred_cxcy, target_cxcy)
    loss_wh = F.mse_loss(pred_wh, target_wh)
    loss_cls = F.binary_cross_entropy_with_logits(pred_cls, target_cls)
    loss_conf = F.binary_cross_entropy_with_logits(pred_conf, target_conf)
    losses = loss_xy + loss_wh + loss_cls + loss_conf

    return losses

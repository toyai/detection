"""Common utilites to YOLO v3."""

import torch
from ignite.utils import manual_seed
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import box_convert

manual_seed(666)


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


def yolo_loss(pred_bbox, pred_cls, pred_conf, target: Tensor, stride, anchors):
    # target_bbox = torch.zeros_like(pred_bbox, device=pred_bbox.device)
    # target_cls = torch.zeros_like(pred_cls, device=pred_cls.device)
    # target_conf = torch.zeros_like(pred_conf, device=pred_conf.device)

    # target_cxcy, target_wh = torch.split(target_bbox, (2, 2), dim=-1)

    pred_cxcy, pred_wh = torch.split(pred_bbox, (2, 2), dim=-1)
    (
        ious_scores,
        obj_mask,
        target_cxcy,
        target_wh,
        target_cls,
        target_conf,
    ) = build_targets(pred_bbox, pred_cls, pred_conf, target, stride, anchors)

    # for batch_idx, target in enumerate(targets):
    #     target = torch.tensor(target, dtype=pred_cxcy.dtype, device=pred_cxcy.device)
    #     # target needs to be * grid_size / img_size
    #     target = target * pred_bbox.size(2) / img_size
    #     target[:, :-1] = box_convert(target[:, :-1], "xyxy", "cxcywh")
    #     cxcy_t, wh_t, cls_t = torch.split(target, (2, 2, 1), dim=-1)
    #     ious = torch.stack([wh_bbox_iou(anchor, wh_t) for anchor in scaled_anchors])
    #     best_ious, best_n = ious.max(0)
    #     height, width = cxcy_t.long().t()

    #     target_cxcy[batch_idx, best_n, height, width, :] = cxcy_t - cxcy_t.floor()
    #     target_wh[batch_idx, best_n, height, width, :] = torch.log(
    #         wh_t / scaled_anchors[best_n] + 1e-16
    #     )
    #     target_conf[batch_idx, best_n, height, width, :] = 1
    #     target_cls[batch_idx, best_n, height, width, cls_t.long()] = 1

    loss_xy = F.mse_loss(pred_cxcy, target_cxcy)
    loss_wh = F.mse_loss(pred_wh, target_wh)
    loss_cls = F.binary_cross_entropy_with_logits(pred_cls, target_cls)
    loss_conf = F.binary_cross_entropy_with_logits(pred_conf, target_conf)
    losses = loss_xy + loss_wh + loss_cls + loss_conf

    return losses


def build_targets(pred_bbox, pred_cls, pred_conf, target: Tensor, stride, anchors):
    target_bbox = torch.zeros_like(pred_bbox, device=pred_bbox.device)
    target_cls = torch.zeros_like(pred_cls, device=pred_cls.device)
    ious_scores = torch.zeros_like(pred_cls, device=pred_cls.device)
    obj_mask = torch.zeros_like(pred_conf, device=pred_conf.device)

    target_cxcy, target_wh = torch.split(target_bbox, (2, 2), dim=-1)
    target[:, 2:6] = box_convert(target[:, 2:6], "xyxy", "cxcywh") / stride

    cxcywh_t = torch.narrow(target, dim=-1, start=2, length=4)
    cxcy_t = torch.narrow(cxcywh_t, dim=-1, start=0, length=2)
    wh_t = torch.narrow(cxcywh_t, dim=-1, start=2, length=2)
    ious = torch.stack([wh_bbox_iou(anchor, wh_t) for anchor in anchors])
    best_ious, best_n = torch.max(ious, 0)

    batch, labels = torch.narrow(target, dim=-1, start=0, length=2).long().t()
    width, height = cxcy_t.long().t()

    obj_mask[batch, best_n, height, width, :] = 1

    target_cxcy[batch, best_n, height, width, :] = cxcy_t - torch.floor(cxcy_t)
    target_wh[batch, best_n, height, width, :] = torch.log(
        wh_t / anchors[best_n] + 1e-16
    )

    target_cls[batch, best_n, height, width, labels] = 1
    # ious_scores[batch, best_n, height, width] = box_iou(
    #     pred_bbox[batch, best_n, height, width], target_bbox
    # )

    target_conf = obj_mask.float()

    return ious_scores, obj_mask, target_cxcy, target_wh, target_cls, target_conf


# def yolo_loss(pred: Tensor, target: Tensor = None):
#     threshold = Threshold(0.5, 0)
#     pred[..., -1] = threshold(pred[..., -1])
#     cls = torch.narrow(pred, -1, 4, 20)
#     # print(cls, cls.shape)
#     cls, cls_idx = torch.max(cls, -1)
#     # print(cls, cls.shape, cls_idx)
#     cxcywh, _, conf = torch.split(pred, (4, 20, 1), dim=-1)
#     pred = torch.cat((cxcywh, cls.unsqueeze(-1), conf), dim=-1)
#     print(pred[..., -1])
#     non_zero = torch.nonzero(pred[..., -1])
#     print(non_zero, non_zero.shape)
#     print(pred.shape, pred)


# net = YOLOv3(416, 20)
# x = torch.rand(1, 3, 416, 416)
# pred = net(x)

# yolo_loss(pred)

# pred 1, 10647, 25
# target 1, num_obj, 6
# pred_cxcy
# pred_wh
# 9 anchors htl ga best 1 anchor ko shr, but how?
# iou between 9 anchors and pred_wh

# def yolo_loss(pred, target):
#     pred_cxcy, pred_wh, pred_cls, pred_conf = torch.split(pred, (2, 2, 20, 1), -1)

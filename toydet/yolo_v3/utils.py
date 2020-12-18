"""Common utilites to YOLO v3."""

import torch
from ignite.utils import manual_seed
from torch import Tensor
from torch.nn import functional as F
from torch.nn.functional import pad
from torchvision.ops import box_convert, box_iou

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


def yolo_loss(pred, target: Tensor, stride, anchors):
    # target_bbox = torch.zeros_like(pred_bbox, device=pred_bbox.device)
    # target_cls = torch.zeros_like(pred_cls, device=pred_cls.device)
    # target_conf = torch.zeros_like(pred_conf, device=pred_conf.device)

    # target_cxcy, target_wh = torch.split(target_bbox, (2, 2), dim=-1)

    # pred_cxcy, pred_wh = torch.split(pred, (2, 2), dim=2)
    # (
    #     ious_scores,
    #     obj_mask,
    #     target_cxcy,
    #     target_wh,
    #     target_cls,
    #     target_conf,
    # ) = build_targets(pred, target, stride, anchors)
    pred_x, pred_y, pred_w, pred_h, pred_conf, pred_cls = pred
    (
        ious_scores,
        obj_mask,
        target_x,
        target_y,
        target_w,
        target_h,
        target_conf,
        target_cls,
    ) = build_targets(pred, target, stride, anchors)

    loss_x = F.mse_loss(pred_x[obj_mask], target_x[obj_mask])
    loss_y = F.mse_loss(pred_y[obj_mask], target_y[obj_mask])
    loss_w = F.mse_loss(pred_w[obj_mask], target_w[obj_mask])
    loss_h = F.mse_loss(pred_h[obj_mask], target_h[obj_mask])
    loss_cls = F.cross_entropy(pred_cls.permute(0, 2, 1, 3, 4), target_cls)
    loss_conf = F.binary_cross_entropy_with_logits(
        pred_conf[obj_mask], target_conf[obj_mask]
    )
    losses = loss_x + loss_y + loss_w + loss_h + loss_cls + loss_conf
    return (loss_x, loss_y, loss_w, loss_h, loss_cls, loss_conf), losses, ious_scores

    # loss_xy = F.mse_loss(pred_cxcy, target_cxcy)
    # loss_wh = F.mse_loss(pred_wh, target_wh)
    # loss_cls = F.binary_cross_entropy_with_logits(pred_cls, target_cls)
    # loss_conf = F.binary_cross_entropy_with_logits(pred_conf, target_conf)
    # losses = loss_xy + loss_wh + loss_cls + loss_conf

    # return (loss_xy, loss_wh, loss_cls, loss_conf), losses


def build_targets(pred, target: Tensor, stride, anchors):
    pred_x, pred_y, pred_w, pred_h, pred_conf, pred_cls = pred
    target_x = torch.zeros_like(pred_x, device=pred_x.device)
    target_y = torch.zeros_like(pred_y, device=pred_y.device)
    target_w = torch.zeros_like(pred_w, device=pred_w.device)
    target_h = torch.zeros_like(pred_h, device=pred_h.device)
    target_conf = torch.zeros_like(pred_conf, device=pred_conf.device)
    target_cls = torch.zeros(
        (pred_cls.shape[0], pred_cls.shape[1], pred_cls.shape[3], pred_cls.shape[4]),
        device=pred_cls.device,
        dtype=torch.long,
    )

    # target_bbox = torch.zeros_like(pred_bbox, device=pred_bbox.device)
    # target_cls = torch.zeros_like(pred_cls, device=pred_cls.device)
    ious_scores = torch.zeros_like(pred_cls, device=pred_cls.device)
    obj_mask = torch.zeros_like(pred_conf, device=pred_conf.device, dtype=torch.long)

    # target_cxcy, target_wh = torch.split(target_bbox, (2, 2), dim=2)
    target_ = target.clone()
    target_[:, 2:6] = box_convert(target[:, 2:6], "xyxy", "cxcywh") / stride
    # assert target_ >= 0.0

    cxcy_t = torch.narrow(target_, dim=-1, start=2, length=2)
    wh_t = torch.narrow(target_, dim=-1, start=4, length=2)
    wh_t_ = pad(wh_t, (2, 0, 0, 0))
    anchors_ = pad(anchors, (2, 0, 0, 0))
    ious = box_iou(anchors_, wh_t_)
    best_ious, best_n = torch.max(ious, 0)

    batch, labels = torch.narrow(target, dim=-1, start=0, length=2).long().t()
    width, height = cxcy_t.long().t()

    obj_mask[batch, best_n, :, height, width] = 1

    target_x[batch, best_n, :, height, width] = (
        cxcy_t[:, 0] - torch.floor(cxcy_t[:, 0])
    ).unsqueeze(-1)
    target_y[batch, best_n, :, height, width] = (
        cxcy_t[:, 1] - torch.floor(cxcy_t[:, 1])
    ).unsqueeze(-1)
    target_w[batch, best_n, :, height, width] = (
        torch.log(wh_t[:, 0] / anchors[best_n][:, 0])
    ).unsqueeze(-1)
    target_h[batch, best_n, :, height, width] = (
        torch.log(wh_t[:, 1] / anchors[best_n][:, 1])
    ).unsqueeze(-1)
    target_cls[batch, best_n, height, width] = labels
    # target_bbox = torch.cat(
    #     (
    #         target_x[batch, best_n, :, height, width],
    #         target_y[batch, best_n, :, height, width],
    #         target_w[batch, best_n, :, height, width],
    #         target_h[batch, best_n, :, height, width],
    #     ),
    #     dim=-1,
    # )
    # pred_bbox = torch.cat(
    #     (
    #         pred_x[batch, best_n, :, height, width],
    #         pred_y[batch, best_n, :, height, width],
    #         pred_w[batch, best_n, :, height, width],
    #         pred_h[batch, best_n, :, height, width],
    #     ),
    #     dim=-1,
    # )
    # ious_scores[batch, best_n, :, height, width] = box_iou(pred_bbox, target_bbox)

    # target_cxcy[batch, best_n, :, height, width] = cxcy_t - torch.floor(cxcy_t)
    # target_wh[batch, best_n, :, height, width] = torch.log(
    #     wh_t / (anchors[best_n] + 1e-16)
    # )

    # target_cls[batch, best_n, labels, height, width] = 1
    # # ious_scores[batch, best_n, height, width] = box_iou(
    # #     pred_bbox[batch, best_n, height, width], target_bbox
    # # )

    target_conf = obj_mask.float()

    # return ious_scores, obj_mask, target_cxcy, target_wh, target_cls, target_conf
    return (
        ious_scores,
        obj_mask,
        target_x,
        target_y,
        target_w,
        target_h,
        target_conf,
        target_cls,
    )

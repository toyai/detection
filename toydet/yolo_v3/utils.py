"""Common utilites to YOLO v3."""

import torch
from torch import Tensor
from torchvision.ops import box_convert


def box_iou_wh(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def yolo_loss(pred, target: Tensor, stride, anchors):
    pred_bbox, pred_conf, pred_cls = pred

    obj_mask, target_bbox, target_conf, target_cls = build_targets(
        pred, target, stride, anchors
    )

    loss_xywh = F.mse_loss(
        torch.masked_select(pred_bbox, obj_mask),
        torch.masked_select(target_bbox, obj_mask),
    )
    loss_cls = F.cross_entropy(pred_cls.permute(0, 2, 1, 3, 4), target_cls)
    loss_conf = F.binary_cross_entropy_with_logits(
        torch.masked_select(pred_conf, obj_mask),
        torch.masked_select(target_conf, obj_mask),
    )

    return loss_xywh, loss_conf, loss_cls


def get_abs_yolo_bbox(t_bbox: Tensor, anchors: Tensor):
    # t_bbox: [batch, num_anchors, grid_size, grid_size, 4]
    t_xy, t_wh = torch.split(t_bbox, (2, 2), dim=-1)
    aranged_tensor = torch.arange(t_bbox.size(2), device=t_bbox.device)
    # grid_x is increasing values in y-axis (same value in x-axis)
    # grid_y is increasing values in x-axis (same value in y-axis)
    # --------
    # grid_x: [grid_size, grid_size]
    # tensor([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    # [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
    # [ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2],
    # [ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
    # [ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
    # [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5],
    # [ 6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6],
    # [ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7],
    # [ 8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8],
    # [ 9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9],
    # [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    # [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11],
    # [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]])
    # --------
    # grid_y: [grid_size, grid_size]
    # tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]])
    # ---------
    grid_x, grid_y = torch.meshgrid(aranged_tensor, aranged_tensor)
    # grid_xy: [1, 1, grid_size, grid_size, 2]
    grid_xy = torch.stack((grid_y, grid_x), dim=-1)[None, None, :, :, :]
    # grid_wh: [1, 3, 1, 1, 2]
    # tensor([[[[[ 3.6250,  2.8125]]],
    #      [[[ 4.8750,  6.1875]]],
    #      [[[11.6562, 10.1875]]]]])
    grid_wh = anchors[None, :, None, None, :]
    # bx = sigmoid(tx) + cx
    # by = sigmoid(ty) + cy
    b_xy = torch.add(t_xy, grid_xy)
    # bw = exp(tw) * pw
    # bh = exp(th) * ph
    b_wh = torch.exp(t_wh) * grid_wh
    return torch.cat((b_xy, b_wh), dim=-1)


def get_rel_yolo_bbox(b_bbox: Tensor, anchors: Tensor):
    # b_bbox is already scaled to grid_size
    b_xy, b_wh = torch.split(b_bbox, (2, 2), dim=-1)
    t_xy = b_xy - torch.floor(b_xy)
    t_wh = torch.log(b_wh / anchors[None, :, None, None, :])
    return torch.cat((t_xy, t_wh), dim=-1)


def build_targets(
    pred: Tensor, target: Tensor, anchors: Tensor, ignore_threshold: float = 0.5
):
    pred_bbox, pred_conf, pred_cls = torch.split(pred, (4, 1, 20), dim=-1)
    target_bbox = torch.zeros_like(pred_bbox)
    target_cls = torch.zeros(
        (pred_cls.size(0), pred_cls.size(1), pred_cls.size(2), pred_cls.size(3)),
        dtype=torch.long,
    )
    # ious_scores = torch.zeros_like(pred_cls, device=pred_cls.device)
    obj_mask = torch.zeros_like(pred_conf, dtype=torch.bool)
    noobj_mask = torch.ones_like(pred_conf, dtype=torch.bool)

    cxcy_t = torch.narrow(target, dim=-1, start=2, length=2)
    wh_t = torch.narrow(target, dim=-1, start=4, length=2)
    ious = torch.stack([box_iou_wh(anchor, wh_t) for anchor in anchors], dim=0)
    _, idx = torch.max(ious, 0)
    batch, labels = torch.narrow(target, dim=-1, start=0, length=2).long().t()
    width, height = cxcy_t.long().t()

    # obj_mask = 1 => there is obj
    # obj_mask = 0 => there is no obj
    obj_mask[batch, idx, height, width, :] = 1
    # noobj_mask = 1 => there is no obj
    # noobj_mask = 0 => there is obj
    noobj_mask[batch, idx, height, width, :] = 0
    for i, iou in enumerate(ious.t()):
        noobj_mask[batch[i], iou > ignore_threshold, height[i], width[i], :] = 0

    target_bbox[batch, idx, height, width, :2] = cxcy_t - torch.floor(cxcy_t)
    target_bbox[batch, idx, height, width, 2:] = torch.log(wh_t / anchors[idx] + 1e-16)
    target_cls[batch, idx, height, width] = labels
    # ious_scores[batch, idx, :, height, width] = box_iou(pred_bbox, target_bbox)

    target_conf = obj_mask.float()
    pred_bbox = torch.masked_select(pred_bbox, obj_mask)
    pred_obj = torch.masked_select(pred_conf, obj_mask)
    pred_noobj = torch.masked_select(pred_conf, noobj_mask)
    target_bbox = torch.masked_select(target_bbox, obj_mask)
    target_obj = torch.masked_select(target_conf, obj_mask)
    target_noobj = torch.masked_select(target_conf, noobj_mask)

    pred = {
        "bbox": pred_bbox,
        "cls": pred_cls.permute(0, 4, 1, 2, 3),
        "obj": pred_obj,
        "noobj": pred_noobj,
    }
    target = {
        "bbox": target_bbox,
        "cls": target_cls,
        "obj": target_obj,
        "noobj": target_noobj,
    }
    return pred, target

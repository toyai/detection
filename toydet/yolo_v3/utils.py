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


def get_abs_yolo_bbox(t_bbox: Tensor, anchors):
    # t_bbox: [batch, num_classes + 5, num_anchors, grid_size, grid_size]
    grid_size = t_bbox.size(-1)
    t_xy, t_wh = torch.split(t_bbox, (2, 2), dim=1)
    aranged_tensor = torch.arange(grid_size, device=t_bbox.device)
    # grid_x is increasing values in x-axis
    # grid_y is increasing values in y-axis
    grid_y, grid_x = torch.meshgrid(aranged_tensor, aranged_tensor)
    # grid_xy: [1, 2, 1, grid_size, grid_size]
    grid_xy = torch.stack((grid_x, grid_y), dim=0)[None, :, None, :, :]
    # grid_wh: [1, 2, 3, 1, 1]
    grid_wh = anchors.t()[None, :, :, None, None]

    b_xy = torch.add(t_xy, grid_xy)
    b_wh = torch.exp(t_wh) * grid_wh
    b_bbox = torch.cat((b_xy, b_wh), dim=1)

    return b_bbox


def get_rel_yolo_bbox(b_bbox: Tensor, anchors):
    b_xy, b_wh = torch.split(b_bbox, (2, 2), dim=-1)
    t_xy = b_xy - torch.floor(b_xy)
    t_wh = torch.log(b_wh / anchors)

    return torch.cat((t_xy, t_wh), dim=-1)


def build_targets(pred, target: Tensor, stride, anchors):
    # pred_bbox, pred_conf, pred_cls = pred
    pred_bbox, pred_conf, pred_cls = torch.split(pred, (4, 1, 20), dim=1)

    target_bbox = torch.zeros_like(pred_bbox, device=pred_bbox.device)
    # target_conf = torch.zeros_like(pred_conf, device=pred_conf.device)
    target_cls = torch.zeros(
        (pred_cls.shape[0], pred_cls.shape[2], pred_cls.shape[3], pred_cls.shape[4]),
        device=pred_cls.device,
        dtype=torch.long,
    )

    # ious_scores = torch.zeros_like(pred_cls, device=pred_cls.device)
    obj_mask = torch.zeros_like(pred_conf, device=pred_conf.device, dtype=torch.bool)

    target_ = target.clone()
    target_[:, 2:6] = box_convert(target[:, 2:6], "xyxy", "cxcywh") / stride
    # assert target_ >= 0.0
    cxcy_t = torch.narrow(target_, dim=-1, start=2, length=2)
    wh_t = torch.narrow(target_, dim=-1, start=4, length=2)
    ious = torch.stack([box_iou_wh(anchor, wh_t) for anchor in anchors], dim=0)
    _, best_n = torch.max(ious, 0)
    batch, labels = torch.narrow(target, dim=-1, start=0, length=2).long().t()
    width, height = cxcy_t.long().t()

    obj_mask[batch, :, best_n, height, width] = 1

    target_bbox[batch, :2, best_n, height, width] = cxcy_t - torch.floor(cxcy_t)
    target_bbox[batch, 2:, best_n, height, width] = torch.log(wh_t / anchors[best_n])
    target_cls[batch, best_n, height, width] = labels

    # ious_scores[batch, best_n, :, height, width] = box_iou(pred_bbox, target_bbox)

    target_conf = obj_mask.float()

    pred_bbox = torch.masked_select(pred_bbox, obj_mask)
    pred_conf = torch.masked_select(pred_conf, obj_mask)
    target_bbox = torch.masked_select(target_bbox, obj_mask)
    target_conf = torch.masked_select(target_conf, obj_mask)

    pred = {
        "bbox": pred_bbox,
        "cls": pred_cls,
        "conf": pred_conf,
    }
    target = {"bbox": target_bbox, "cls": target_cls, "conf": target_conf}
    return pred, target

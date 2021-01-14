"""Common utilites to YOLO v3."""

import torch
from torch import Tensor


def box_iou_wh(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def get_abs_yolo_bbox(t_bbox: Tensor, anchors: Tensor):
    # t_bbox: [batch, num_anchors, grid_size, grid_size, 4]
    t_xy, t_wh = torch.split(t_bbox, (2, 2), dim=-1)
    size = t_bbox.size(2)
    aranged_tensor = torch.arange(size, device=t_bbox.device)
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
    grid_xy = torch.stack((grid_y, grid_x), dim=-1).reshape(1, 1, size, size, 2)
    # grid_wh: [1, 3, 1, 1, 2] - anchors.reshape(1, 3, 1, 1, 2)
    # tensor([[[[[ 3.6250,  2.8125]]],
    #      [[[ 4.8750,  6.1875]]],
    #      [[[11.6562, 10.1875]]]]])
    # bx = sigmoid(tx) + cx
    # by = sigmoid(ty) + cy
    b_xy = torch.add(t_xy, grid_xy)
    # bw = exp(tw) * pw
    # bh = exp(th) * ph
    b_wh = torch.exp(t_wh) * anchors.reshape(1, 3, 1, 1, 2)
    b_wh = torch.where(
        torch.logical_or(torch.isnan(b_wh), torch.isinf(b_wh)),
        torch.tensor(0.0, device=b_wh.device),
        b_wh,
    )
    return torch.cat((b_xy, b_wh), dim=-1)

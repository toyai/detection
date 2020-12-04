"""Common utilites to YOLO v3."""
import torch


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

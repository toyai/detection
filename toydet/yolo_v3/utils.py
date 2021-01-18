"""Common utilites to YOLO v3."""

import torch
from torch import Tensor
from torchvision.ops import nms, box_convert
from typing import List


def parse_config(name: str) -> List[dict]:
    """Parse YOLOv3 config from given name.

    Args:
        name (str): name of YOLOv3 config. [yolov3, yolov3_voc, yolov3_tiny, yolov3_spp]

    Returns:
        list[dict]: configs
    """
    if name == "yolov3":
        from toydet.yolo_v3 import YOLOV3 as CONFIG
    elif name == "yolov3_voc":
        from toydet.yolo_v3 import YOLOV3_VOC as CONFIG
    elif name == "yolov3_tiny":
        from toydet.yolo_v3 import YOLOV3_TINY as CONFIG
    elif name == "yolov3_spp":
        from toydet.yolo_v3 import YOLOV3_SPP as CONFIG
    else:
        raise ValueError(
            f"{name} is undefined. Choose between"
            " (yolov3, yolov3_voc, yolov3_tiny, yolov3_spp)"
        )

    lines = CONFIG.split("\n")
    lines = [line.strip() for line in lines if line and not line.startswith("#")]
    configs = []
    for line in lines:
        if line.startswith("["):
            configs.append({})
            configs[-1]["type"] = line.strip("[]")
            if configs[-1]["type"] == "convolutional":
                configs[-1]["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            configs[-1][key.strip()] = value.strip()

    return configs


def box_iou_wh(wh1: Tensor, wh2: Tensor) -> Tensor:
    """Find IoU between ``wh1`` and ``wh2``. ``wh1`` and ``wh2`` are
    assumed to have the same centroid.

    Args:
        wh1 (Tensor): Tensor containing width and height.
        wh2 (Tensor): Tensor containing width and height.

    Returns:
        Tensor: IoU
    """
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def non_max_suppression(preds, conf_threshold, nms_threshold):
    preds[..., :4] = box_convert(preds[..., :4], "cxcywh", "xyxy")
    output = [None for _ in range(len(preds))]
    for i, pred in enumerate(preds):
        pred = pred[pred[:, 4] > conf_threshold]
        if not pred.size(0):
            continue
        score = pred[:, 4] * pred[:, 5:].max(1)[0]
        pred = pred[(-score).argsort()]
        class_conf, class_idx = pred[:, 5:].max(1, keepdim=True)
        pred = torch.cat((pred[:, :5], class_conf, class_idx), dim=-1)
        keep = nms(pred[:, :4], score, nms_threshold)
        # if keep is not None:
        output[i] = pred[keep]
    return output

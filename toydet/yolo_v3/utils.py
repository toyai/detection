"""Common utilites to YOLO v3."""

from typing import List

import torch
from torch import Tensor
from torchvision import ops as box_ops


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


def postprocess_predictions(preds, config):
    # convert cxcywh to xyxy
    preds[..., :4] = box_ops.box_convert(preds[..., :4], "cxcywh", "xyxy")
    pred_boxes_list, pred_scores_list, pred_cls_list = preds.split(
        (4, 1, config.num_classes), dim=-1
    )
    all_boxes = []
    all_labels = []
    all_scores = []

    for boxes, scores, labels in zip(pred_boxes_list, pred_scores_list, pred_cls_list):
        boxes = box_ops.clip_boxes_to_image(boxes, (config.img_size, config.img_size))
        scores = scores.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > config.conf_threshold)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
        if boxes.size(0) and scores.size(0) and labels.size(0):
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            class_prob, labels = labels.max(-1)
            scores *= class_prob
            keep = box_ops.batched_nms(boxes, scores, labels, config.nms_threshold)
            keep = keep[: config.detections_per_img]
            boxes, scores, labels = (boxes[keep], scores[keep], labels[keep])

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)

    return all_boxes, all_labels, all_scores

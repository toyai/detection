"""
train_fn and evaluate_fn used in train.py.
"""

import torch

from toydet.yolo_v3.utils import yolo_loss

# --------------------
# training function
# --------------------


def train_fn(batch, net, optimizer, device):
    net.train(True)
    img, target = batch[0].to(device), batch[1].to(device)
    out1, out2, out3 = net(img)
    loss_xywh_1, loss_conf_1, loss_cls_1 = yolo_loss(
        out1,
        target,
        net.neck.block1.yolo_layer.stride,
        net.neck.block1.yolo_layer.scaled_anchors,
    )
    loss_xywh_2, loss_conf_2, loss_cls_2 = yolo_loss(
        out2,
        target,
        net.neck.block2.yolo_layer.stride,
        net.neck.block2.yolo_layer.scaled_anchors,
    )
    loss_xywh_3, loss_conf_3, loss_cls_3 = yolo_loss(
        out3,
        target,
        net.neck.block3.yolo_layer.stride,
        net.neck.block3.yolo_layer.scaled_anchors,
    )
    losses = (
        loss_xywh_1
        + loss_conf_1
        + loss_cls_1
        + loss_xywh_2
        + loss_conf_2
        + loss_cls_2
        + loss_xywh_3
        + loss_conf_3
        + loss_cls_3
    )
    loss_item = {
        "stride/0": net.neck.block1.yolo_layer.stride,
        "stride/1": net.neck.block2.yolo_layer.stride,
        "stride/2": net.neck.block2.yolo_layer.stride,
        "loss/xywh/0": loss_xywh_1,
        "loss/xywh/1": loss_xywh_2,
        "loss/xywh/2": loss_xywh_3,
        "loss/cls/0": loss_cls_1,
        "loss/cls/1": loss_cls_2,
        "loss/cls/2": loss_cls_3,
        "loss/conf/0": loss_conf_1,
        "loss/conf/1": loss_conf_2,
        "loss/conf/2": loss_conf_3,
        "loss/total": losses,
        # "ious/0": ious_scores_1,
        # "ious/1": ious_scores_2,
        # "ious/2": ious_scores_3,
    }
    # loss_item["stride"] = (
    #     net.neck.block1.yolo_layer.stride,
    #     net.neck.block2.yolo_layer.stride,
    #     net.neck.block3.yolo_layer.stride,
    # )
    # loss_item["loss_xywh"] = (
    #     loss_xywh_1.detach().cpu().item(),
    #     loss_xywh_2.detach().cpu().item(),
    #     loss_xywh_3.detach().cpu().item(),
    # )
    # loss_item["loss_cls"] = (
    #     loss_cls_1.detach().cpu().item(),
    #     loss_cls_2.detach().cpu().item(),
    #     loss_cls_3.detach().cpu().item(),
    # )
    # loss_item["loss_conf"] = (
    #     loss_conf_1.detach().cpu().item(),
    #     loss_conf_2.detach().cpu().item(),
    #     loss_conf_3.detach().cpu().item(),
    # )
    # loss_item["ious"] = (
    #     ious_scores_1.detach().cpu().item(),
    #     ious_scores_2.detach().cpu().item(),
    #     ious_scores_3.detach().cpu().item(),
    # )
    # loss_item["losses"] = losses.detach().cpu().item()

    losses.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss_item


# ---------------------
# evaluation function
# ---------------------


@torch.no_grad()
def evaluate_fn(batch, net, optimizer, device):
    net.eval()
    img, target = batch[0].to(device), batch[1].to(device)
    out1, out2, out3 = net(img)
    loss_xywh_1, loss_conf_1, loss_cls_1 = yolo_loss(
        out1,
        target,
        net.neck.block1.yolo_layer.stride,
        net.neck.block1.yolo_layer.scaled_anchors,
    )
    loss_xywh_2, loss_conf_2, loss_cls_2 = yolo_loss(
        out2,
        target,
        net.neck.block2.yolo_layer.stride,
        net.neck.block2.yolo_layer.scaled_anchors,
    )
    loss_xywh_3, loss_conf_3, loss_cls_3 = yolo_loss(
        out3,
        target,
        net.neck.block3.yolo_layer.stride,
        net.neck.block3.yolo_layer.scaled_anchors,
    )
    losses = (
        loss_xywh_1
        + loss_conf_1
        + loss_cls_1
        + loss_xywh_2
        + loss_conf_2
        + loss_cls_2
        + loss_xywh_3
        + loss_conf_3
        + loss_cls_3
    )
    loss_item = {
        "stride/0": net.neck.block1.yolo_layer.stride,
        "stride/1": net.neck.block2.yolo_layer.stride,
        "stride/2": net.neck.block2.yolo_layer.stride,
        "loss/xywh/0": loss_xywh_1,
        "loss/xywh/1": loss_xywh_2,
        "loss/xywh/2": loss_xywh_3,
        "loss/cls/0": loss_cls_1,
        "loss/cls/1": loss_cls_2,
        "loss/cls/2": loss_cls_3,
        "loss/conf/0": loss_conf_1,
        "loss/conf/1": loss_conf_2,
        "loss/conf/2": loss_conf_3,
        "loss/total": losses,
        # "ious/0": ious_scores_1,
        # "ious/1": ious_scores_2,
        # "ious/2": ious_scores_3,
    }
    # loss_item["stride"] = (
    #     net.neck.block1.yolo_layer.stride,
    #     net.neck.block2.yolo_layer.stride,
    #     net.neck.block3.yolo_layer.stride,
    # )
    # loss_item["loss_xywh"] = (
    #     loss_xywh_1.detach().cpu().item(),
    #     loss_xywh_2.detach().cpu().item(),
    #     loss_xywh_3.detach().cpu().item(),
    # )
    # loss_item["loss_cls"] = (
    #     loss_cls_1.detach().cpu().item(),
    #     loss_cls_2.detach().cpu().item(),
    #     loss_cls_3.detach().cpu().item(),
    # )
    # loss_item["loss_conf"] = (
    #     loss_conf_1.detach().cpu().item(),
    #     loss_conf_2.detach().cpu().item(),
    #     loss_conf_3.detach().cpu().item(),
    # )
    # loss_item["ious"] = (
    #     ious_scores_1.detach().cpu().item(),
    #     ious_scores_2.detach().cpu().item(),
    #     ious_scores_3.detach().cpu().item(),
    # )
    # loss_item["losses"] = losses.detach().cpu().item()

    return loss_item

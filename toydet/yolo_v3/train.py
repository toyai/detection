import os
import logging
from argparse import ArgumentParser
from datetime import datetime
from random import randint, randrange

import ignite.distributed as idist
import torch
from ignite.contrib.handlers import WandBLogger
from ignite.engine import Engine, Events

# from ignite.metrics import Loss
from ignite.utils import manual_seed, setup_logger
from torch import optim
from torchvision.ops import box_convert, nms, batched_nms
from torchvision.transforms.functional import to_pil_image

import wandb
from toydet.transforms import (
    LetterBox,
    MultiArgsSequential,
    RandomHorizontalFlip_,
    RandomVerticalFlip_,
)
from toydet.utils import cuda_info, draw_bounding_boxes, mem_info
from toydet.yolo_v3 import models
from toydet.yolo_v3.datasets import CLASSES, VOCDetection_, get_dataloader

manual_seed(666)

parser = ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--model_name", default="yolov3_darknet53_voc", type=str)
parser.add_argument("--max_epochs", default=10, type=int)
parser.add_argument("--verbose", "-v", action="store_true", help="use logging.INFO")
parser.add_argument("--filepath", default=None, type=str, help="write stdout to file")
parser.add_argument("--log_train", type=int, default=50)
parser.add_argument("--log_eval", type=int, default=1)
# parser.add_argument("--amp", action="store_true")
parser.add_argument("--sanity_check", default=1, type=int, help="0 to disable")
parser.add_argument("--overfit_batches", type=int, default=0)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--plot_img", action="store_true")

config = parser.parse_args()

if config.verbose:
    config.verbose = logging.INFO
else:
    config.verbose = logging.WARNING

# logger
logger = setup_logger(
    format="[%(levelname)s] %(message)s",
    level=config.verbose,
    filepath=config.filepath,
)

# ---------------------------------
# model, optimizer, device, loss
# ---------------------------------
device = idist.device()
net = getattr(models, config.model_name)().to(device)
optimizer = optim.Adam(net.parameters(), lr=config.lr)

# -----------------
# train function
# -----------------


def train_fn(engine: Engine, batch):
    net.train(True)
    img, target = batch[0].to(device), batch[1].to(device)
    losses_list = net(img, target)
    losses = sum(loss["loss/total"] for loss in losses_list)
    # pred1, target1 = build_targets(
    #     pred1,
    #     target,
    #     net.neck.block1.yolo_layer.stride,
    #     net.neck.block1.yolo_layer.scaled_anchors,
    # )
    # pred2, target2 = build_targets(
    #     pred2,
    #     target,
    #     net.neck.block2.yolo_layer.stride,
    #     net.neck.block2.yolo_layer.scaled_anchors,
    # )
    # pred3, target3 = build_targets(
    #     pred3,
    #     target,
    #     net.neck.block3.yolo_layer.stride,
    #     net.neck.block3.yolo_layer.scaled_anchors,
    # )
    # loss_xywh_1 = mse_loss(pred1["bbox"], target1["bbox"])
    # loss_cls_1 = ce_loss(pred1["cls"], target1["cls"])
    # loss_conf_1 = bce_loss_with_logits(pred1["conf"], target1["conf"])
    # loss_xywh_2 = mse_loss(pred2["bbox"], target2["bbox"])
    # loss_cls_2 = ce_loss(pred2["cls"], target2["cls"])
    # loss_conf_2 = bce_loss_with_logits(pred2["conf"], target2["conf"])
    # loss_xywh_3 = mse_loss(pred3["bbox"], target3["bbox"])
    # loss_cls_3 = ce_loss(pred3["cls"], target3["cls"])
    # loss_conf_3 = bce_loss_with_logits(pred3["conf"], target3["conf"])
    # losses = (
    #     loss_xywh_1
    #     + loss_conf_1
    #     + loss_cls_1
    #     + loss_xywh_2
    #     + loss_conf_2
    #     + loss_cls_2
    #     + loss_xywh_3
    #     + loss_conf_3
    #     + loss_cls_3
    # )
    loss_item = {
        "epoch": engine.state.epoch,
        # "stride/0": net.neck.block1.yolo_layer.stride,
        # "stride/1": net.neck.block2.yolo_layer.stride,
        # "stride/2": net.neck.block3.yolo_layer.stride,
        "loss/xywh/0": losses_list[0]["loss/xywh"],
        "loss/xywh/1": losses_list[1]["loss/xywh"],
        "loss/xywh/2": losses_list[2]["loss/xywh"],
        "loss/cls/0": losses_list[0]["loss/cls"],
        "loss/cls/1": losses_list[1]["loss/cls"],
        "loss/cls/2": losses_list[2]["loss/cls"],
        "loss/conf/0": losses_list[0]["loss/conf"],
        "loss/conf/1": losses_list[1]["loss/conf"],
        "loss/conf/2": losses_list[2]["loss/conf"],
        "loss/total": losses.detach().cpu().item(),
        # "ious/0": ious_scores_1,
        # "ious/1": ious_scores_2,
        # "ious/2": ious_scores_3,
    }

    losses.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {k: v for k, v in sorted(loss_item.items())}


# -------------------
# evaluate function
# -------------------


@torch.no_grad()
def evaluate_fn(engine: Engine, batch, conf_threshold: float = 0.5):
    net.eval()
    img, target = batch[0].to(device), batch[1].to(device)
    preds = net(img)
    preds = torch.cat(preds, dim=-1)
    conf_mask = (preds[:, 4, :] > conf_threshold).float().unsqueeze(1)
    preds = preds * conf_mask
    os.makedirs("./predictions", exist_ok=True)
    for i, pred in enumerate(preds):
        pred = pred.t()
        boxes = box_convert(pred[:, :4], "cxcywh", "xyxy")
        scores, idxs = torch.max(pred[:, 5:], 1)
        # idxs = torch.randint(0, 20, (10647,))
        keep = batched_nms(boxes, scores, idxs, 0.5)
        _, box_idx = torch.max(boxes[keep], 0)
        best_cls, _ = torch.max(pred[:, 5:][keep], 1)
        labels = [CLASSES[int(label)] for label in best_cls.tolist()]
        img_ = draw_bounding_boxes(img[i], boxes[keep][box_idx], labels)
        img_.save(f"./predictions/{datetime.now().isoformat()}.png", format="png")
    # pred_bbox = box_convert(pred[:, :4, :].reshape(-1, 4), "cxcywh", "xyxy")
    # ious = box_iou(pred_bbox, target[:, 2:6])
    # best_ious, best_n = torch.max(ious, 0)
    # # print(best_ious)
    # cls = pred[:, 5:, :].reshape(-1, 20)
    # best_cls, best_cls_n = torch.max(cls, 1)
    # labels = [CLASSES[int(label)] for label in best_cls.tolist()]
    # img = draw_bounding_boxes(img.squeeze(0), pred_bbox[best_n], labels)
    # img.show()
    return pred, target


# -----------------------
# train and eval engine
# -----------------------
engine_train = Engine(train_fn)
engine_eval = Engine(evaluate_fn)


# Loss(
#     mse_loss,
#     lambda output: (output["pred1"]["bbox"], output["target1"]["bbox"]),
#     device=device,
# ).attach(engine_eval, "loss/xywh/0")
# Loss(
#     ce_loss,
#     lambda output: (output["pred1"]["cls"], output["target1"]["cls"]),
#     device=device,
# ).attach(engine_eval, "loss/cls/0")
# Loss(
#     bce_loss_with_logits,
#     lambda output: (output["pred1"]["conf"], output["target1"]["conf"]),
#     device=device,
# ).attach(engine_eval, "loss/conf/0")
# Loss(
#     mse_loss,
#     lambda output: (output["pred2"]["bbox"], output["target2"]["bbox"]),
#     device=device,
# ).attach(engine_eval, "loss/xywh/1")
# Loss(
#     ce_loss,
#     lambda output: (output["pred2"]["cls"], output["target2"]["cls"]),
#     device=device,
# ).attach(engine_eval, "loss/cls/1")
# Loss(
#     bce_loss_with_logits,
#     lambda output: (output["pred2"]["conf"], output["target2"]["conf"]),
#     device=device,
# ).attach(engine_eval, "loss/conf/1")
# Loss(
#     mse_loss,
#     lambda output: (output["pred3"]["bbox"], output["target3"]["bbox"]),
#     device=device,
# ).attach(engine_eval, "loss/xywh/2")
# Loss(
#     ce_loss,
#     lambda output: (output["pred3"]["cls"], output["target3"]["cls"]),
#     device=device,
# ).attach(engine_eval, "loss/cls/2")
# Loss(
#     bce_loss_with_logits,
#     lambda output: (output["pred3"]["conf"], output["target3"]["conf"]),
#     device=device,
# ).attach(engine_eval, "loss/conf/2")

# --------------------------
# train and eval transforms
# --------------------------
transforms_train = MultiArgsSequential(
    LetterBox(416), RandomHorizontalFlip_(), RandomVerticalFlip_()
)
transforms_eval = MultiArgsSequential(LetterBox(416))

# ---------------------------
# train and eval dataloader
# ---------------------------
if config.sanity_check:  # for sanity checking
    dataloader_eval = get_dataloader(VOCDetection_, 2, "val", transforms_eval)
    engine_train.add_event_handler(
        Events.STARTED,
        lambda: engine_eval.run(dataloader_eval, max_epochs=1, epoch_length=2),
    )
if config.overfit_batches:  # for overfitting
    dataloader_train = get_dataloader(
        VOCDetection_, config.overfit_batches, "train", transforms_train, overfit=True
    )
    engine_train.add_event_handler(
        Events.EPOCH_COMPLETED, lambda: engine_eval.run(dataloader_train, max_epochs=1)
    )
else:
    dataloader_train = get_dataloader(
        VOCDetection_, config.batch_size, "train", transforms_train
    )
    dataloader_eval = get_dataloader(
        VOCDetection_, config.batch_size, "val", transforms_eval
    )
    engine_train.add_event_handler(
        Events.EPOCH_COMPLETED, lambda: engine_eval.run(dataloader_eval, max_epochs=1)
    )

# -------------------------
# plot transformed images
# -------------------------


@engine_train.on(Events.ITERATION_STARTED(every=config.log_train))
def plot_transformed_imgs(engine):
    img, target = engine.state.batch[0], engine.state.batch[1].numpy()
    idx = randrange(len(img))
    img = img[idx]
    mask = target[..., 0] == idx
    target = target[mask]

    if config.wandb and config.plot_img:
        box_data = []
        for t in target:
            class_id = int(t[1].item())
            box_data.append(
                {
                    "position": {
                        "minX": t[2].item() / img.shape[-1],
                        "minY": t[3].item() / img.shape[-2],
                        "maxX": t[4].item() / img.shape[-1],
                        "maxY": t[5].item() / img.shape[-2],
                    },
                    "class_id": class_id,
                    "box_caption": CLASSES[class_id],
                }
            )
        boxes = {
            "transforms": {
                "box_data": box_data,
                "class_labels": {k: v for k, v in enumerate(CLASSES)},
            }
        }
        wandb.log({"transforms": wandb.Image(to_pil_image(img, "RGB"), boxes=boxes)})
    elif not config.wandb and config.plot_img:
        boxes = target[..., 2:]
        labels = [CLASSES[int(label)] for label in target[..., 1].tolist()]
        colors = [
            (randint(0, 200), randint(0, 200), randint(0, 200))
            for _ in range(len(labels))
        ]
        img = draw_bounding_boxes(img, boxes, labels, colors)
        fig_name = datetime.now().isoformat() + ".png"
        img.save(fig_name, format="png")
        logger.info("Transformed image saved as %s", fig_name)


if config.wandb:
    # --------------
    # wandb logger
    # --------------
    wb_logger = WandBLogger(config=config, name=config.model_name, project="yolov3")
    wb_logger.watch(net, log="gradients", log_freq=config.log_train)

    # --------------------------
    # logging training metrics
    # --------------------------
    wb_logger.attach_output_handler(
        engine_train,
        Events.ITERATION_COMPLETED(every=config.log_train),
        tag="train",
        output_transform=lambda output: output,
    )

    # ----------------------------
    # logging evaluation metrics
    # ----------------------------
    wb_logger.attach_output_handler(
        engine_eval,
        Events.EPOCH_COMPLETED(every=config.log_eval),
        tag="eval",
        metric_names="all",
        global_step_transform=lambda *_: engine_train.state.iteration,
    )


# -----------------------
# log metrics to stdout
# -----------------------
def log_metrics(engine, mode, output):
    logger.info(
        "%s Epoch %i - Iteration %i : %s"
        % (mode, engine.state.epoch, engine.state.iteration, output),
    )


engine_train.add_event_handler(
    Events.ITERATION_COMPLETED(every=config.log_train),
    lambda engine: log_metrics(engine, "Train", engine.state.output),
)

engine_eval.add_event_handler(
    Events.EPOCH_COMPLETED(every=config.log_eval),
    lambda engine: log_metrics(
        engine, "Eval", {k: v for k, v in sorted(engine.state.metrics.items())}
    ),
)

if __name__ == "__main__":
    logger.info("Running on %s ...", device)
    logger.info("Configs %s", config)
    params = sum(p.numel() for p in net.parameters())
    gradients = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("Number of parameters: %g", params)
    logger.info("Number of gradients: %g", gradients)
    # logger.info("Model size: %.4f MB", params * 4 / (1024.0 * 1024.0))
    if "cuda" in device.type:
        name = cuda_info(logger, device)

    # run the training
    engine_train.run(dataloader_train, max_epochs=config.max_epochs)

    if "cuda" in device.type:
        mem_info(logger, device, name)

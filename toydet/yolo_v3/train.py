from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pformat
from typing import Sequence

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.utils import manual_seed
from torch import Tensor, nn, optim
from torchvision.ops import box_convert, nms

from toydet import get_default_parser
from toydet.transforms import (
    LetterBox,
    RandomHorizontalFlipWithBBox,
    RandomVerticalFlipWithBBox,
    SequentialWithDict,
)
from toydet.utils import (
    cuda_info,
    get_dataloaders,
    get_wandb_logger,
    log_metrics,
    params_info,
    sanity_check,
    setup_logging,
)
from toydet.datasets import VOCDetection_
from toydet.yolo_v3.models import YOLOv3
from toydet.yolo_v3.utils import box_iou_wh

torch.autograd.set_detect_anomaly(True)

transforms_train = SequentialWithDict(
    {
        "LetterBox": LetterBox(416),
        "RandomHorizontalFlipWithBBox": RandomHorizontalFlipWithBBox(),
        "RandomVerticalFlipWithBBox": RandomVerticalFlipWithBBox(),
    }
)
transforms_eval = SequentialWithDict({"LetterBox": LetterBox(416)})


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        self.loss_dict = {
            "loss/xywh": 0,
            "loss/conf": 0,
            "loss/cls": 0,
            "loss/total": 0,
        }

    def forward(self, pred, target, anchors, num_classes):
        pred, target = self.make_pred_and_target(pred, target, anchors, num_classes)
        loss_xywh = self.mse_loss(pred[0], target[0])
        loss_conf = self.bce_with_logits_loss(pred[1], target[1])
        loss_cls = self.bce_with_logits_loss(pred[2], target[2])
        losses = loss_xywh + loss_conf + loss_cls
        self.loss_dict["loss/xywh"] += loss_xywh.detach().cpu().item()
        self.loss_dict["loss/conf"] += loss_conf.detach().cpu().item()
        self.loss_dict["loss/cls"] += loss_cls.detach().cpu().item()
        self.loss_dict["loss/total"] += losses.detach().cpu().item()
        return losses

    def make_pred_and_target(self, pred, target, anchors, num_classes):
        # target: [number of objects in a batch, 6]
        # target is also normalized by img_size, so its [0-1]
        # multiplied with grid_size for grid_size x grid_size output
        target = torch.cat((target[..., :2], target[..., 2:] * pred.size(2)), dim=-1)
        pred_bbox, pred_conf, pred_cls = torch.split(pred, (4, 1, num_classes), dim=-1)
        batch, labels, target_xy, target_wh = torch.split(target, (1, 1, 2, 2), dim=-1)
        batch, labels = batch.long().squeeze(-1), labels.long().squeeze(-1)
        width, height = target_wh[..., 0].long(), target_wh[..., 1].long()

        ious = torch.stack([box_iou_wh(anchor, target_wh) for anchor in anchors], dim=0)
        _, iou_idx = torch.max(ious, 0)

        target_xy = target_xy - torch.floor(target_xy)
        target_wh = torch.log(target_wh / anchors[iou_idx])
        pred_bbox = pred_bbox[batch, iou_idx, height, width, :]
        target_bbox = torch.cat((target_xy, target_wh), dim=-1)
        if not pred_bbox.shape == target_bbox.shape:
            raise AssertionError(
                f"Got pred_bbox {pred_bbox.shape}, target_bbox {target_bbox.shape}"
            )
        target_cls = nn.functional.one_hot(labels, num_classes).type_as(pred_cls)

        obj_mask = torch.zeros_like(pred_conf, dtype=torch.bool)
        obj_mask[batch, iou_idx, height, width, :] = 1
        noobj_mask = 1 - obj_mask.float()
        for i, iou in enumerate(ious.t()):
            noobj_mask[batch[i], iou > 0.5, height[i], width[i], :] = 0

        pred_obj = torch.masked_select(pred_conf, obj_mask)
        pred_noobj = torch.masked_select(pred_conf, noobj_mask.to(torch.bool))
        target_obj = torch.ones_like(pred_obj)
        target_noobj = torch.zeros_like(pred_noobj)
        pred_conf = torch.cat((pred_obj, pred_noobj), dim=0)
        target_conf = torch.cat((target_obj, target_noobj), dim=0)
        return (pred_bbox, pred_conf, pred_cls[batch, iou_idx, height, width, :]), (
            target_bbox,
            target_conf,
            target_cls,
        )


def train_fn(
    engine: Engine,
    batch: Sequence[Tensor],
    net: nn.Module,
    optimizer: optim.Optimizer,
    config: Namespace,
) -> dict:
    net.train(True)
    img = batch[0].to(config.device, non_blocking=True)
    target = batch[1].to(config.device, non_blocking=True)
    # make target normalized by img_size, so its [0-1]
    target[..., 2:] = box_convert(target[..., 2:], "xyxy", "cxcywh") / config.img_size
    results = net(img)
    yolo_loss = YOLOLoss()
    losses = 0
    losses += yolo_loss(
        results[0],
        target,
        net.detector.module_dict.yolo_layer_0.anchors,
        config.num_classes,
    )
    losses += yolo_loss(
        results[1],
        target,
        net.detector.module_dict.yolo_layer_1.anchors,
        config.num_classes,
    )
    losses += yolo_loss(
        results[2],
        target,
        net.detector.module_dict.yolo_layer_2.anchors,
        config.num_classes,
    )

    losses.backward()
    optimizer.step()
    optimizer.zero_grad()

    yolo_loss.loss_dict["epoch"] = engine.state.epoch
    return dict(sorted(yolo_loss.loss_dict.items()))


@torch.no_grad()
def evaluate_fn(
    engine: Engine,
    batch: Sequence[Tensor],
    net: nn.Module,
    config: Namespace,
):
    net.eval()
    img = batch[0].to(config.device, non_blocking=True)
    target = batch[1].to(config.device, non_blocking=True)
    preds = net(img)
    preds = torch.cat(preds, dim=1)
    conf_mask = preds[..., 4:5] >= config.conf_threshold
    preds = preds * conf_mask
    preds[..., :4] = box_convert(preds[..., :4] * config.img_size, "cxcywh", "xyxy")
    raw_score = preds[..., 4:5] * preds[..., 5:]
    raw_bbox = preds[..., :4]
    bbox = []
    label = []
    score = []
    for l in range(config.num_classes):
        bbox_l = raw_bbox
        score_l = raw_score[..., l]
        mask = score_l >= config.conf_threshold
        bbox_l = bbox_l[mask]
        score_l = score_l[mask]
        keep = nms(bbox_l, score_l, 0.7)
        bbox_l = bbox_l[keep]
        score_l = score_l[keep]

        if len(bbox_l):
            bbox.append(bbox_l)
            label.append(torch.tensor((l,) * len(bbox_l)))
        if len(score_l):
            score.append(score_l)

    if len(bbox) and len(label) and len(score):
        bbox = torch.vstack(bbox)
        label = torch.hstack(label)
        score = torch.hstack(score)

        max_score, idx = torch.max(score, dim=0)
        label = label[idx]
        bbox = bbox[idx]
        # img = draw_bounding_boxes()
        print(max_score, idx, label, bbox)

    # pred_cls, _ = torch.max(preds[..., 5:], dim=-1, keepdim=True)
    # preds = torch.cat((preds[..., :5], pred_cls), dim=-1)
    # zero_mask = preds != 0
    return preds, target


def main(local_rank: int, config: Namespace):
    """Main function of YOLOv3 training / evaluation"""

    # --------------------------
    # datasets and dataloaders
    # --------------------------
    dataset_eval = VOCDetection_(
        config.data_path, image_set="val", transforms=transforms_eval
    )
    dataset_train = VOCDetection_(config.data_path, transforms=transforms_train)
    dataloader_train, dataloader_eval = get_dataloaders(
        dataset_train, dataset_eval, config
    )

    # --------------------------------------
    # Events of train / eval for logging
    # --------------------------------------
    # pylint:disable=not-callable
    log_train_events = Events.ITERATION_COMPLETED(
        lambda _, event: event % config.log_train == 0 or event == 1
    )
    log_eval_events = Events.EPOCH_COMPLETED(every=config.log_eval)

    # -----------------------
    # model and optimizer
    # -----------------------
    net = idist.auto_model(YOLOv3(config.img_size, config.num_classes))
    optimizer = idist.auto_optim(optim.Adam(net.parameters(), lr=config.lr))

    # ---------------
    # setup logging
    # ---------------
    config.__dict__.update(**optimizer.defaults)
    logger, name = setup_logging(optimizer, config)
    logger.info("Configs\n%s", pformat(vars(config)))
    params_info(net)
    cuda_info(config.device)

    # ---------------------
    # engine train / eval
    # ---------------------
    engine_train = Engine(
        lambda engine, batch: train_fn(engine, batch, net, optimizer, config)
    )
    engine_eval = Engine(lambda engine, batch: evaluate_fn(engine, batch, net, config))

    # ---------------
    # sanity check
    # ---------------
    if config.sanity_check:
        engine_train.add_event_handler(
            Events.STARTED, sanity_check, engine_eval, dataloader_eval, config
        )

    # -------------------
    # add wandb logger
    # -------------------
    if config.wandb:
        wb_logger = get_wandb_logger(
            config, name, engine_train, engine_eval, log_train_events, log_eval_events
        )

    # ----------------
    # log metrics
    # ----------------
    engine_eval.add_event_handler(log_eval_events, log_metrics, "Eval", config.device)
    engine_train.add_event_handler(
        log_train_events, log_metrics, "Train", config.device
    )

    # ---------------------------
    # eval engine run / overfit
    # ---------------------------
    if config.overfit_batches:
        logger.info("Eval overfitting with %i.", config.overfit_batches)
        engine_train.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda: engine_eval.run(
                dataloader_train, max_epochs=1, epoch_length=config.overfit_batches
            ),
        )
    else:
        epoch_length_eval = (
            config.epoch_length_eval
            if isinstance(config.epoch_length_eval, int)
            else round(len(dataloader_eval) * config.epoch_length_eval)
        )
        engine_train.add_event_handler(
            Events.EPOCH_COMPLETED,
            lambda: engine_eval.run(
                dataloader_eval, max_epochs=1, epoch_length=epoch_length_eval
            ),
        )

    # ----------------------------
    # train engine run / overfit
    # ----------------------------
    if config.overfit_batches:
        logger.info("Train overfitting with %i.", config.overfit_batches)
        engine_train.run(
            dataloader_train,
            max_epochs=config.max_epochs,
            epoch_length=config.overfit_batches,
        )
    else:
        epoch_length_train = (
            config.epoch_length_train
            if isinstance(config.epoch_length_train, int)
            else round(len(dataloader_train) * config.epoch_length_train)
        )
        engine_train.run(
            dataloader_train,
            max_epochs=config.max_epochs,
            epoch_length=epoch_length_train,
        )
    if config.wandb:
        wb_logger.finish()


if __name__ == "__main__":
    parser = ArgumentParser(
        "YOLOv3 training and evaluation script", parents=[get_default_parser()]
    )
    parser.add_argument("--img_size", default=416, type=int, help="image size (416)")
    parser.add_argument(
        "--num_classes", default=20, type=int, help="number of classes (20)"
    )
    parser.add_argument(
        "--conf_threshold", default=0.5, type=float, help="confidence threshold (0.5)"
    )
    opt = parser.parse_args()
    manual_seed(opt.seed)
    if opt.filepath:
        path = Path(opt.filepath)
        path.mkdir(parents=True, exist_ok=True)
        opt.filepath = path
    with idist.Parallel(
        idist.backend(), opt.nproc_per_node, opt.nnodes, opt.node_rank
    ) as parallel:
        parallel.run(main, opt)

from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pformat
from typing import Sequence

import ignite.distributed as idist
import torch
from ignite.engine import Engine, Events
from ignite.utils import manual_seed
from torch import Tensor, nn, optim
from torchvision.ops import box_convert

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
from toydet.yolo_v3.datasets import VOCDetection_
from toydet.yolo_v3.models import YOLOv3

transforms_train = SequentialWithDict(
    {
        "LetterBox": LetterBox(416),
        "RandomHorizontalFlipWithBBox": RandomHorizontalFlipWithBBox(),
        "RandomVerticalFlipWithBBox": RandomVerticalFlipWithBBox(),
    }
)
transforms_eval = SequentialWithDict({"LetterBox": LetterBox(416)})


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
    loss_dict, total_loss = net(img, target)

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_dict["epoch"] = engine.state.epoch
    return dict(sorted(loss_dict.items()))


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
    conf_mask = (preds[..., 4] > config.conf_threshold).unsqueeze(-1)
    preds = preds * conf_mask
    preds[..., :4] = box_convert(preds[..., :4] * config.img_size, "cxcywh", "xyxy")
    pred_cls, _ = torch.max(preds[..., 5:], dim=-1, keepdim=True)
    preds = torch.cat((preds[..., :5], pred_cls), dim=-1)
    zero_mask = preds != 0
    return preds, target


def main(local_rank: int, config: Namespace):
    """Main function of YOLOv3 training / evaluation"""

    # --------------------------
    # datasets and dataloaders
    # --------------------------
    dataset_eval = VOCDetection_(
        config.dataset_path, image_set="val", transforms=transforms_eval
    )
    dataset_train = VOCDetection_(config.dataset_path, transforms=transforms_train)
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
    parser.add_argument("--img_size", default=416, type=int, help="image size")
    parser.add_argument("--num_classes", default=20, type=int, help="number of classes")
    parser.add_argument(
        "--conf_threshold", default=0.5, type=float, help="confidence threshold"
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

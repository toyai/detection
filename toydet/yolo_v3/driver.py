"""YOLOv3 Driver."""

# from multiprocessing import cpu_count
from argparse import ArgumentParser

import torch
from ignite.engine import Engine
from ignite.utils import manual_seed
from torch.utils.data import DataLoader

from toydet.transforms import LetterBox, MultiArgsSequential
from toydet.yolo_v3.model import YOLOv3
from toydet.yolo_v3.utils import yolo_loss
from toydet.yolo_v3.voc import VOCDataset, collate_fn

manual_seed(666)

# torch.autograd.set_detect_anomaly(True)


def main(args):
    transforms = MultiArgsSequential(LetterBox(416))

    train_dl = DataLoader(
        VOCDataset(download=False, transforms=transforms, batch_size=args.batch_size),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        # num_workers=cpu_count(),
    )
    # val_dl = DataLoader(
    #     VOCDataset(download=False, transforms=transforms),
    #     batch_size=2,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    # )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"TRAINING ON {DEVICE} ...")
    net = YOLOv3(416, 20).to(DEVICE)
    optim = torch.optim.Adam(net.parameters())

    def train_fn(engine, batch):
        """Train function."""
        net.train()
        imgs, targets = batch
        imgs = imgs.to(DEVICE)
        out1, out2, out3 = net(imgs)
        loss_1 = yolo_loss(
            *out1,
            targets,
            net.neck.block1.yolo_layer.stride,
            net.neck.block1.yolo_layer.scaled_anchors,
        )
        loss_2 = yolo_loss(
            *out2,
            targets,
            net.neck.block2.yolo_layer.stride,
            net.neck.block2.yolo_layer.scaled_anchors,
        )
        loss_3 = yolo_loss(
            *out3,
            targets,
            net.neck.block3.yolo_layer.stride,
            net.neck.block3.yolo_layer.scaled_anchors,
        )
        losses = loss_1 + loss_2 + loss_3
        s_loss = losses.detach().cpu().item()
        optim.zero_grad()
        losses.backward()
        optim.step()
        print("=> Loss: ", s_loss)

        return {"loss": s_loss}

    train_engine = Engine(train_fn)
    train_engine.run(train_dl, max_epochs=1)


if __name__ == "__main__":
    parser = ArgumentParser("YOLOv3 training script.")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)

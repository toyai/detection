"""YOLOv3 Driver."""

# from multiprocessing import cpu_count

import albumentations as A
import torch
from ignite.engine import Engine
from ignite.utils import manual_seed
from torch.utils.data import DataLoader

from jiance.yolo_v3.model import YOLOv3
from jiance.yolo_v3.utils import yolo_loss
from jiance.yolo_v3.voc import VOCDataset, collate_fn

manual_seed(666)

transforms = A.Compose(
    [
        A.Resize(416, 416),
        # A.HorizontalFlip(),
        # A.RandomBrightness(),
        # A.VerticalFlip(),
    ],
    bbox_params=A.BboxParams(format="pascal_voc"),
)

train_dl = DataLoader(
    VOCDataset(download=False, transforms=transforms),
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    # num_workers=cpu_count(),
)
val_dl = DataLoader(
    VOCDataset(download=False, transforms=transforms),
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"TRAINING ON {DEVICE}...")
net = YOLOv3(416, 20).to(DEVICE)
optim = torch.optim.Adam(net.parameters())


def train_fn(engine, batch):
    """Train function."""
    net.train()
    imgs, targets = batch
    imgs = imgs.to(DEVICE)
    out1, out2, out3 = net(imgs)
    loss_1 = yolo_loss(out1, targets, 416, net.neck.block1.yolo_layer.scaled_anchors)
    loss_2 = yolo_loss(out2, targets, 416, net.neck.block2.yolo_layer.scaled_anchors)
    loss_3 = yolo_loss(out3, targets, 416, net.neck.block3.yolo_layer.scaled_anchors)
    losses = loss_1 + loss_2 + loss_3
    s_loss = losses.detach().cpu().item()
    optim.zero_grad()
    losses.backward()
    optim.step()
    print("=> Loss: ", s_loss)

    return {"loss": s_loss}


train_engine = Engine(train_fn)
train_engine.run(train_dl, max_epochs=1)

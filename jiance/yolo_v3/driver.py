from multiprocessing import cpu_count

import albumentations as A
import torch
from ignite.engine import Engine
from ignite.utils import manual_seed
from torch.utils.data import DataLoader

from jiance.yolo_v3.model import YOLOv3
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
    VOCDataset(download=True, transforms=transforms),
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=cpu_count(),
)
val_dl = DataLoader(
    VOCDataset(download=False, transforms=transforms),
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn,
)

net = YOLOv3("jiance/yolo_v3/cfg/yolov3-voc.cfg").cuda()
optim = torch.optim.Adam(net.parameters())


def train_fn(engine, batch):
    imgs, targets = batch
    imgs, targets = imgs.cuda(), targets
    yolo_outputs, loss = net(imgs, targets)
    s_loss = loss.detach().cpu().item()
    optim.zero_grad(True)
    loss.backward()
    optim.step()
    print("=> Loss: ", s_loss)

    return {"loss": s_loss}


engine = Engine(train_fn)
engine.run(train_dl, max_epochs=5)

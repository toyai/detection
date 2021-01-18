"""
Custom object detection with common neural networks.
This module will contain common object detection neural networks
and backbone neural networks.
"""

__version__ = "0.1.0"

import re
from argparse import ArgumentParser

import torch
from ignite.distributed import device


def int_or_float(value):
    if re.match(r"\d+\.\d+", value):
        value = float(value)
    elif re.match(r"\d+", value, re.ASCII):
        value = int(value)

    return value


DEFAULTS = {
    "amp": {
        "action": "store_true",
        "help": "to use torch.cuda.amp",
    },
    "batch_size": {
        "default": 2,
        "type": int,
        "help": "will be equally divided by number of GPUs if in distributed (2)",
    },
    "data_path": {
        "default": "./datasets",
        "type": str,
        "help": "datasets path (.)",
    },
    "device": {
        "default": device(),
        "type": torch.device,
        "help": "device to use for training / evaluation / testing (idist.device())",
    },
    "drop_last": {
        "action": "store_true",
        "help": "set to True to drop the last incomplete batch",
    },
    "epoch_length_train": {
        "default": 1.0,
        "type": int_or_float,
        "help": """epoch_length of ignite.Engine.run() [int, float] if float,
                    round(epoch_length * len(dataloader)) if int, epoch_length (1.0)""",
    },
    "epoch_length_eval": {
        "default": 1.0,
        "type": int_or_float,
        "help": """epoch_length of ignite.Engine.run() [int, float] if float,
                    round(epoch_length * len(dataloader)) if int, epoch_length (1.0)""",
    },
    "filepath": {
        "default": "./logs",
        "type": str,
        "help": "logging file path (./logs)",
    },
    "j": {
        "default": 0,
        "type": int,
        "help": "num_workers for DataLoader (0)",
    },
    "max_epochs": {
        "default": 2,
        "type": int,
        "help": "max_epochs of ignite.Engine.run() (2)",
    },
    "lr": {
        "default": 1e-3,
        "type": float,
        "help": "learning rate used by torch.optim.* (1e-3)",
    },
    "log_train": {
        "default": 50,
        "type": int,
        "help": "logging interval of training iteration (50)",
    },
    "log_eval": {
        "default": 1,
        "type": int,
        "help": "logging interval of evaluation epoch (1)",
    },
    "overfit_batches": {
        "default": 0,
        "type": int,
        "help": "try overfitting the model (0)",
    },
    "save_img": {
        "action": "store_true",
        "help": "save the transformed image",
    },
    "sanity_check": {
        "default": 2,
        "type": int,
        "help": "sanity checking the evaluation first in batches (2)",
    },
    "seed": {
        "default": 666,
        "type": int,
        "help": "used in ignite.utils.manual_seed() (666)",
    },
    "wandb": {
        "action": "store_true",
        "help": "to use wandb or not",
    },
    "verbose": {
        "action": "store_true",
        "help": "Use logging.INFO",
    },
    "nproc_per_node": {
        "default": None,
        "type": int,
        "help": """The number of processes to launch on each node, for GPU training
                this is recommended to be set to the number of GPUs in your system
                so that each process can be bound to a single GPU (None)""",
    },
    "nnodes": {
        "default": None,
        "type": int,
        "help": "The number of nodes to use for distributed training (None)",
    },
    "node_rank": {
        "default": None,
        "type": int,
        "help": "The rank of the node for multi-node distributed training (None)",
    },
}


def get_default_parser():
    """Get the default configs for training."""
    parser = ArgumentParser(add_help=False)

    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", **value)

    return parser

"""
Custom object detection with common neural networks.
This module will contain common object detection neural networks
and backbone neural networks.
"""

__version__ = "0.1.0"


from argparse import ArgumentParser
from typing import Union

DEFAULTS = {
    "amp": {
        "action": "store_true",
        "help": "to use torch.cuda.amp",
    },
    "batch_size": {
        "default": 2,
        "type": int,
        "help": "will be equally divided by number of GPUs if in distributed",
    },
    "epoch_length_train": {
        "default": 1.0,
        "type": Union[float, int],
        "help": """epoch_length of ignite.Engine.run() [int, float] if float,
                    round(epoch_length * len(dataloader)) if int, epoch_length""",
    },
    "epoch_length_eval": {
        "default": 1.0,
        "type": Union[float, int],
        "help": """epoch_length of ignite.Engine.run() [int, float] if float,
                    round(epoch_length * len(dataloader)) if int, epoch_length""",
    },
    "j": {
        "default": 0,
        "type": int,
        "help": "num_workers for DataLoader",
    },
    "max_epochs": {
        "default": 2,
        "type": int,
        "help": "max_epochs of ignite.Engine.run()",
    },
    "lr": {
        "default": 1e-3,
        "type": float,
        "help": "learning rate used by torch.optim.*",
    },
    "log_train": {
        "default": 50,
        "type": int,
        "help": "logging interval of training iteration",
    },
    "log_eval": {
        "default": 1,
        "type": int,
        "help": "logging interval of evaluation epoch",
    },
    "overfit_batches": {
        "default": 0,
        "type": int,
        "help": "try overfitting the model",
    },
    "path": {
        "default": "./datasets",
        "type": str,
        "help": "datasets path",
    },
    "save_img": {
        "default": False,
        "type": bool,
        "help": "save the transformed image",
    },
    "sanity_check": {
        "default": 2,
        "type": int,
        "help": "sanity checking the evaluation first in batches",
    },
    "seed": {
        "default": 666,
        "type": int,
        "help": "used in ignite.utils.manual_seed()",
    },
    "wandb": {
        "default": False,
        "type": bool,
        "help": "to use wandb or not",
    },
    "nproc_per_node": {
        "default": None,
        "type": int,
        "help": "The number of processes to launch on each node, for GPU training this is recommended to be set to the number of GPUs in your system so that each process can be bound to a single GPU",
    },
    "nnodes": {
        "default": None,
        "type": int,
        "help": "The number of nodes to use for distributed training",
    },
    "node_rank": {
        "default": None,
        "type": int,
        "help": "The rank of the node for multi-node distributed training",
    },
}


def get_default_parser():
    """Get the default configs for training."""
    parser = ArgumentParser(add_help=False)

    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", **value)

    return parser


get_default_parser()

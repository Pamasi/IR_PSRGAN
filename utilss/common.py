import argparse
import sys
from os import getcwd, mkdir, listdir
import os.path as osp
import numpy as np
import torch
import wandb
from typing import Any, Dict, Optional, Callable, Optional, Tuple

import wandb.wandb_run

            
def save_ckpt(
    net: torch.nn.Module,
    epoch: int,
    loss: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.CyclicLR] = None,
    dir: str = '',
    torch_state='',
    save_best: bool = False
) -> None:

    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr': scheduler.state_dict() if scheduler is not None else None,
                'torch_state': torch_state
                },
               get_ckpt_dir('best' if save_best else epoch, dir)
               )


def load_ckpt(
    ckpt_dir: str,
    net: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.CyclicLR] = None
) -> int:

    ckpt = torch.load(ckpt_dir)

    net.load_state_dict(ckpt['model_state_dict'])

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(['lr'])

    torch.set_rng_state(ckpt['torch_state'])


    epoch = ckpt['epoch']

    return epoch


def get_ckpt_dir(epoch: int, dir: str = '') -> str:
    if osp.exists(dir) == False:
        mkdir(osp.join(getcwd(), dir))

    return osp.join(getcwd(), f'{dir}/ckpt_{epoch}', )


def config_wandb(config: Dict[str, Any]) -> Tuple[wandb.wandb_run.Run, str]:
    print(sys.executable)
    wandb.login()
    

    wandb_tag = [f'{param}@{val}' for param, val in vars(config).items()]

    wandb_run = wandb.init(project='ir_psrgan', config=config, reinit=True, tags=wandb_tag)


    return wandb_run


def listdir_filter(path: str):
    filtered_listdir = []
    for f in listdir(path):
        if not f.startswith('.'):
            filtered_listdir.append(f)
    return filtered_listdir

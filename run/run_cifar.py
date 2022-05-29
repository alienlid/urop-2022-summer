import argparse
import itertools
from pathlib import Path
import os
import random
import time
import copy 
import datetime as dt
from pprint import pprint

import numpy as np 
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from collections import defaultdict, Counter

from src.data import datasets
from src.data import transforms as data_transforms
from src.data import ffcv_pipelines

from src.models import cifar_resnet
from src.models import multihead

from src.train import trainer

from src.utils import data_utils
from src.utils import train_utils 
from src.utils import eval_utils 
from src.utils import common_utils 

from src.utils import run_utils
from fastargs import Section, Param, get_current_config
from fastargs.validation import And, OneOf, InRange

"""
train standard erm models on cifar{10,100}
"""

MODEL_ARCHS  = {
    'resnet9': cifar_resnet.ResNet9,
    'resnet18': cifar_resnet.ResNet18,
    'resnet34': cifar_resnet.ResNet34,
    'resnet50': cifar_resnet.ResNet50
}

def get_model(args):
    num_classes = 10 if args.data.dataset=='cifar10' else 100
    arch = args.model.arch 
    assert arch in MODEL_ARCHS
    model = MODEL_ARCHS[arch](num_classes=num_classes)

    if args.dataloading.use_ffcv:
        model = model.to(memory_format=torch.channels_last)

    return model

def get_optimizer(params, args):
    opt_kw = args.optimizer
    opt = optim.SGD(params, lr=opt_kw.lr, weight_decay=opt_kw.wd, momentum=opt_kw.momentum)
    if args.optimizer.use_lr_schedule:
        milestones = [n*opt_kw.lr_decay_gap for n in range(1,opt_kw.num_decay+1)]
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=opt_kw.lr_decay_factor)
    else:
        sch = None
    return opt, sch

def get_data(args):
    data_dir = os.path.expanduser(f'~/datasets/betons/{args.data.dataset}/')
    get_path = lambda split: Path(data_dir) / f'{args.data.dataset}_{split}.beton'
    get_pipelines = lambda split,aug,device: ffcv_pipelines.get_pipelines(args.data.dataset, aug, device)
    get_loader = lambda split,aug,indices,device: data_utils.get_ffcv_loader(get_path(split), 
                                                                             args.dataloading.batch_size,
                                                                             args.dataloading.num_workers,
                                                                             get_pipelines(split, aug, device), 
                                                                             split=='train',
                                                                             os_cache=args.dataloading.os_cache,
                                                                             seed=args.data.seed,
                                                                             indices=indices)

    indices = None 
    if args.dataloading.subsample_prob < 1.0:
        mask = np.random.rand(50_000) < args.dataloading.subsample_prob 
        indices = np.nonzero(mask)[0]

    return {
        'train': get_loader('train', args.expt.aug_scheme, indices, args.training.device),
        'test': get_loader('test', 'test', None, args.training.device)
    }

def evaluate_model(eval_dls, model, device):
    model = model.to(device).eval()
    stats = {}
    for n, dl in eval_dls.items():
        stats[n] = eval_utils.get_accuracy_and_loss(model, dl, device)
    model = model.cpu()
    return stats    
    
def train_model(train_dl, eval_dls, args):
    device = args.training.device
    model = get_model(args)
    model = model.to(device)
    opt, sch = get_optimizer(model.parameters(), args)

    train_kw = args.training 
    tr= trainer.SingleHeadTrainer(model, opt, sch, device, 
                                  save_dir=args.training.save_dir,
                                  enable_logging=args.training.enable_logging,
                                  checkpoint_type=args.training.checkpoint_type)

    stats = tr.train(train_dl, eval_dls,                     
                     args.training.model_selection_loader, 
                     train_stop_crit=train_kw.train_stop_crit, 
                     model_selection_crit=train_kw.model_selection_crit, 
                     max_train_epochs=train_kw.max_train_epochs,
                     train_stop_epsilon=train_kw.train_stop_epsilon,
                     eval_epoch_gap=train_kw.eval_epoch_gap)

    model = stats['checkpoints']['best']
    return model, stats


def run_once(args):
    # setup 
    if args.training.save_root_dir:
        root_dir = Path(os.path.expanduser(args.training.save_root_dir))
        dir_name = f'{args.expt.expt_name}:{train_utils.get_timestamp()}'
        save_dir = root_dir / dir_name 
        save_dir.mkdir(parents=False, exist_ok=True)
        train_utils.BCOLORS.print(f'SAVE_DIR: {save_dir}', 'header')
    else:
        save_dir = None
        
    args.training.save_dir = save_dir
    args.training.device = torch.device(args.training.device_id)

    # load dataset
    eval_dls = get_data(args)
    tr_dl = eval_dls.pop('train')

    # train + save  model
    model, train_stats = train_model(tr_dl, eval_dls, args)

    # save metadata
    args_map = common_utils.convert_namespace_to_dict(args)
    run_utils.pickle_obj(args_map, args.training.save_dir, 'metadata.pkl')

    # save eval
    eval_stats = evaluate_model(eval_dls, model, args.training.device)
    run_utils.pickle_obj(eval_stats, args.training.save_dir, 'eval.pkl')

    return model, eval_stats, args 

def run_sbatch(args):
    if args.expt.sbatch_expt == 'lr':
        lrs = [0.001, 0.005, 0.01, 0.025, 0.05]
        args.optimizer.lr = lr = lrs[args.expt.sbatch_jobid]
        sch = args.optimizer.use_lr_schedule
        if lr < 0.01: sch = args.optimizer.use_lr_schedule = 0
        print (f'LR:{lr}, AUG:{args.expt.aug_scheme}, SCHEDULE:{sch}')
        run_cli(args)

    elif args.expt.sbatch_expt == 'lr_bs':
        lrs = [1e-3, 5e-3, 1e-2, 5e-2]
        bss = [256, 512, 1024, 2048]
        combs = list(itertools.product(lrs, bss))
        combs = [combs[i::4] for i in range(4)][args.expt.sbatch_jobid] 
        
        for idx, (lr, bs) in enumerate(combs,1):
            train_utils.BCOLORS.print(f'COMB {idx}/{len(combs)}', 'header')
            args.dataloading.batch_size = bs
            args.optimizer.lr = lr
            sch = args.optimizer.use_lr_schedule
            if lr < 0.01: sch = args.optimizer.use_lr_schedule = 0
            run_cli(args)

    else:
        pass

def run_cli(args):
    root_dir = Path(os.path.expanduser(args.training.save_root_dir))
    if not root_dir.exists():
        root_dir.mkdir(parents=False, exist_ok=True)
        train_utils.BCOLORS.print(f'MKDIR: {root_dir.as_posix()}', 'header')

    save_dirs = []
    for run_idx in range(args.expt.num_runs):
        train_utils.BCOLORS.print(f'Run {run_idx+1}/{args.expt.num_runs}', 'header')
        *_, args = run_once(args)

        if args.training.save_dir:  
            save_dirs.append(args.training.save_dir.as_posix())
    
    pprint(save_dirs)
        
    return save_dirs

def get_fastargs_sections():
    sections = {
        'data': run_utils.get_data_fastargs_section(),
        'model': run_utils.get_model_fastargs_section(),
        'optimizer': run_utils.get_optimizer_fastargs_section(),
        'dataloading': run_utils.get_dataloading_fastargs_section(),
        'training': run_utils.get_training_fastargs_section()
    }

    sections['expt'] = Section('expt', 'experiment-specific arguments').params(
        expt_name=Param(str, 'experiment name', required=True),
        num_runs=Param(int, 'number of runs', required=True),
        aug_scheme=Param(str, 'aug scheme', default='train'),
        sbatch=Param(And(int, OneOf([0,1])), 'cli/0 or sbatch/1', default=0),
        sbatch_jobid=Param(int, 'slurm array job id'),
        sbatch_expt=Param(str, 'sbatch experiment name')
    )

    return sections 

if __name__=='__main__':
    # get args 
    sections = get_fastargs_sections()
    config = get_current_config()
    parser = argparse.ArgumentParser(description='CIFAR{10,100}')
    config.augment_argparse(parser)
    config.validate(mode='stderr')
    config.summary()

    args = config.get()

    # run
    if args.expt.sbatch:
        run_sbatch(args)
    else:
        run_cli(args)

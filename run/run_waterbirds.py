import argparse
import itertools
from pathlib import Path
import os
import random
from re import L
import time
import copy
import datetime as dt
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from collections import defaultdict, Counter

from tqdm import tqdm

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
train standard erm models on waterbirds
"""

def get_model(args):
    resnets = {
        'resnet18': torchvision.models.resnet18,
        'resnet34': torchvision.models.resnet34,
        'resnet50': torchvision.models.resnet50
    }

    arch = args.model.arch
    pretrained = bool(args.model.pretrained)

    assert arch.lower() in resnets
    backbone = resnets[arch](pretrained=pretrained)
    fc_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    head = nn.Linear(in_features=fc_dim, out_features=2)
    model = multihead.SingleHeadModel(backbone, head, squeeze=True)

    if args.dataloading.use_ffcv:
        model = model.to(memory_format=torch.channels_last)

    return model

def get_data(args):
    data_dir = os.path.expanduser(f'~/datasets/betons/{args.data.dataset}/')
    get_path = lambda split: Path(data_dir) / f'{args.data.dataset}_{split}.beton'
    get_pipelines = lambda split,aug,device: ffcv_pipelines.get_pipelines('waterbirds', aug, device)
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
        mask = np.random.rand(4_795) < args.dataloading.subsample_prob
        indices = np.nonzero(mask)[0]

    return {
        'train': get_loader('train', args.expt.aug_scheme, indices, args.training.device),
        'val': get_loader('val', 'test', None, args.training.device),
        'test': get_loader('test', 'test', None, args.training.device)
    }

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
        sbatch_jobid=Param(int, 'slrum array job id'),
        sbatch_expt=Param(str, 'sbatch experiment name')
    )

    return sections

def get_optimizer(params, args):
    opt_kw = args.optimizer
    opt = optim.SGD(params, lr=opt_kw.lr, weight_decay=opt_kw.wd, momentum=opt_kw.momentum)
    if args.optimizer.use_lr_schedule:
        milestones = [n*opt_kw.lr_decay_gap for n in range(1,opt_kw.num_decay+1)]
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=opt_kw.lr_decay_factor)
    else:
        sch = None
    return opt, sch


def train_model(train_dl, eval_dls, args):
    device = args.training.device
    model = get_model(args)
    model = model.to(device)
    opt, sch = get_optimizer(model.parameters(), args)

    train_kw = args.training
    tr= trainer.SingleHeadTrainerWorstGroupTracker(model, opt, sch, device,
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

def _evaluate_model(dl_name, dl, model, device):
    in_tr_mode = model.training
    model = model.eval()

    wg_meters = defaultdict(train_utils.AverageMeter)

    with torch.no_grad():
        for xb, yb, mb in dl:
            xb = xb.to(device, non_blocking=True)
            yb, mb = map(lambda z: z.clone().cpu().numpy(), [yb, mb])

            with autocast(enabled=True):
                logits = model(xb)

            preds = logits.argmax(-1).cpu().numpy()
            is_correct = (preds==yb).astype(int)

            for is_c, y, m in zip(is_correct, yb, mb):
                wg_meters[(y,m)].update(is_c, 1)

    if in_tr_mode:
        model = model.train()

    group_accs = {k:v.mean() for k,v in wg_meters.items()}
    acc_wg = min(group_accs.values())

    out = {
        'acc_wg': acc_wg,
        'group_accs': group_accs
    }

    print (dl_name.upper())
    pprint(out)

    return out

def evaluate_model(eval_dls, model, device):
    model = model.to(device).eval()
    stats = {}
    for n, dl in eval_dls.items():
        # stats[n] = _evaluate_model(n, model, args)
        stats[n] = _evaluate_model(n, dl, model, device)
    model = model.cpu()
    return stats

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

    # save eval of best model
    model = train_stats['checkpoints']['best'].eval()
    eval_stats = evaluate_model(eval_dls, model, args.training.device)
    run_utils.pickle_obj(eval_stats, args.training.save_dir, f'eval.pkl')

    return model, eval_stats, args

def run_sbatch(args):
    if args.expt.sbatch_expt == 'redo_all':
        NUM_JOBS = 5
        archs = ['resnet50', 'resnet18']
        augs = ['train', 'train_creager']
        bss = [64, 128]
        lrs = [0.001, 0.005]
        pts = [1]
        max_epochs = [30, 50]
        combs = list(itertools.product(archs, augs, bss, lrs, pts, max_epochs))
        combs = [combs[i::NUM_JOBS] for i in range(NUM_JOBS)][args.expt.sbatch_jobid % NUM_JOBS]

        for idx, (arch, aug, bs, lr, pt, max_epoch) in enumerate(combs, 1):
            train_utils.BCOLORS.print(f'COMB {idx}/{len(combs)}', 'header')
            args.model.arch = arch
            args.expt.aug_scheme = aug
            args.dataloading.batch_size = bs
            args.optimizer.lr = lr
            args.model.pretrained = pt
            args.training.max_train_epochs = max_epoch
            run_cli(args)

    elif args.expt.sbatch_expt == 'final':
        args.model.arch = ['resnet50', 'resnet18'][args.expt.sbatch_jobid % 2]
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
        save_dirs.append(args.training.save_dir.as_posix())

    pprint(save_dirs)

    return save_dirs


if __name__=='__main__':
    # get args
    sections = get_fastargs_sections()
    config = get_current_config()
    parser = argparse.ArgumentParser(description='Waterbirds')
    config.augment_argparse(parser)
    config.validate(mode='stderr')
    config.summary()
    args = config.get()

    # run
    if args.expt.sbatch: run_sbatch(args)
    else: run_cli(args)


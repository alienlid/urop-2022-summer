import os
import random 
import torch
from fastargs import Section, Param, get_current_config
from fastargs.validation import And, OneOf, InRange

def pickle_obj(obj, save_dir, filename):
    if not save_dir: return
    assert filename.endswith('.pkl')
    path = os.path.join(save_dir, filename)
    torch.save(obj, path)

def get_data_fastargs_section():
    return Section('data', 'data arguments').params(
        dataset=Param(And(str, OneOf(['cifar10', 'cifar100', 'waterbirds'])), 'dataset'),
        seed=Param(int, 'seed', default=random.randint(0,10_000)),
    )

def get_model_fastargs_section():
    return Section('model', 'model arguments').params(
        arch=Param(str, 'model architecture name'),
        pretrained=Param(And(int, OneOf([0,1])), 'pretrained model')
    )

def get_optimizer_fastargs_section():
    return Section('optimizer', 'optimizer arguments').params(
        lr=Param(float, 'learning rate'),
        wd=Param(float, 'weight decay'),
        momentum=Param(float, 'momentum'),
        use_lr_schedule=Param(And(int, OneOf([0,1])), 'use lr schedule', default=1),
        num_decay=Param(int, 'decay learning rate `num_decay` times', default=2),
        lr_decay_gap=Param(int, 'epoch gap for lr decay'),
        lr_decay_factor=Param(float, 'lr decay factor'),
    )

def get_dataloading_fastargs_section():
    return Section('dataloading', 'dataloader arguments').params(
        batch_size=Param(int, 'batch size'),
        num_workers=Param(int, 'num workers', required=True),
        pin_memory=Param(And(int, OneOf([0,1])), 'pin memory', default=1),
        shuffle=Param(And(int, OneOf([0,1])), 'shuffle', default=1),
        use_ffcv=Param(And(int, OneOf([0,1])), 'use ffcv', default=1),
        os_cache=Param(And(int, OneOf([0,1])), 'os cache', default=1),
        subsample_prob=Param(float, 'probability of including each datapoint', default=1.0)
    )

def get_training_fastargs_section():
    return Section('training', 'training arguments').params(
        model_selection_loader=Param(str, 'dataset to use for model selection'),
        model_selection_crit=Param(str, 'training stat to use for model selection'),
        max_train_epochs=Param(int, 'max training epochs'),
        train_stop_epsilon=Param(float, 'train stop criterion epsilon'),
        train_stop_crit=Param(str, 'train stop criterion'),
        eval_epoch_gap=Param(int, 'epoch gap for model evaluation'),
        device_id=Param(And(int, InRange(min=0)), 'gpu device id', required=True),
        enable_logging=Param(And(int, OneOf([0,1])), 'enable logging', default=0),
        save_root_dir=Param(str, 'save root dir', required=True),
        checkpoint_type=Param(And(str, OneOf(['model', 'state_dict'])), 'checkpoint type'),
    )

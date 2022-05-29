import pathlib
import os
import sys
import copy
import random
import time
import itertools
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn.functional as F
import torchvision

import wilds
from wilds.common.data_loaders import get_train_loader as get_wilds_train_loader
from wilds.common.data_loaders import get_eval_loader as get_wilds_eval_loader

from ..utils import data_utils

from ffcv.fields import IntField, RGBImageField
from ffcv.writer import DatasetWriter

PARENT_DIR = pathlib.Path(__file__).parent
DATA_DIR = os.path.expanduser('~/datasets/')

"""
Classes
- DatasetwithLoadear
- GroupedDataset

- CIFAR
- CorruptCIFAR10
- Waterbirds
"""

class DatasetWithLoader(Dataset):

    def get_loader(self, **loader_kws):
        loader = DataLoader(self, **loader_kws)
        return loader

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        raise NotImplementedError

class GroupedDataset(DatasetWithLoader):

    def __init__(self, dataset, class_to_group_map, replace_original_class, class_batch_index=1):
        """
        Partition data into groups of classes / superclasses
        - dataset: instance of torch.utils.data.Dataset
                with dataset[idx] = (..., label)

        - class_to_group_map: maps class index to group / superclass index 
                            (delete classes that are not in class_to_group_map)

        - replace_original_class: replace original class if true else add group to data tuple 
        - class_map_index: index of label/class (that needs to be mapped) in the batch tuple
        """ 
        # asserts
        groups = set(class_to_group_map.values())
        assert isinstance(dataset, Dataset)
        assert min(groups) == 0 and max(groups) == len(groups)-1
        
        # setup
        self.dataset = dataset
        self.class_to_group_map = class_to_group_map
        self.replace_original_class = replace_original_class
        self.class_batch_index = class_batch_index

        self.num_groups = len(groups)
        self.num_classes = len(self.class_to_group_map)

        # map class subset to [0,..,k]
        old_classes = list(class_to_group_map.keys())
        self.old_to_new_class_map = dict(zip(old_classes, range(self.num_classes)))

        # group to new class map
        self.group_to_class_map = defaultdict(list)
        for c, g in self.class_to_group_map.items():
            c_new = self.old_to_new_class_map[c]
            self.group_to_class_map[g].append(c_new)

        # delete datapoints with classes not in class_to_group map 
        valid_classes = set(self.class_to_group_map)
        valid_indices = []

        for idx, tup in enumerate(self.dataset):
            label = tup[self.class_batch_index]
            if label in valid_classes:
                valid_indices.append(idx)
        
        # note: subset-ing retains transforms / target transforms
        self.dataset = Subset(self.dataset, valid_indices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)): raise IndexError
        data_tuple = list(self.dataset[idx])
        y_old = data_tuple[self.class_batch_index] 

        # group label 
        g = self.class_to_group_map[y_old]

        if self.replace_original_class:
            data_tuple[self.class_batch_index] = g 
            return data_tuple

        # old_to_new label map
        data_tuple[self.class_batch_index] = self.old_to_new_class_map[y_old]
        data_tuple.append(g)

        return data_tuple

class IndexedDataset(DatasetWithLoader):
    """Dataset wrapper that appends datapoint indices to datapoint tuple"""    
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        tup = self.dataset[index]
        return *tup, index

class CIFAR(DatasetWithLoader):
    """Common Dataset class for CIFAR10 and CIFAR100"""

    CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
    CIFAR10_STD = np.array([0.2023, 0.1994, 0.2010])

    CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408])
    CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761])

    def __init__(self, cifar10, train,
                transform=None, target_transform=None):
        super().__init__()
        self.is_cifar10 = cifar10
        self.train = train
        self.split_name = 'train' if self.train else 'test'
        self.transform=transform
        self.target_transform=target_transform

        if self.is_cifar10:
            self.dset_name = 'cifar10'
            self.num_classes = 10
            self.root = os.path.join(DATA_DIR, 'cifar10')
            self.tensor_channel_mean = CIFAR.CIFAR10_MEAN
            self.tensor_channel_std = CIFAR.CIFAR10_STD
            dset_func = torchvision.datasets.CIFAR10
        else:
            self.dset_name = 'cifar100'
            self.num_classes = 100
            self.root = os.path.join(DATA_DIR, 'cifar100')
            self.tensor_channel_mean = CIFAR.CIFAR100_MEAN
            self.tensor_channel_std = CIFAR.CIFAR100_STD
            dset_func = torchvision.datasets.CIFAR100

        self.dataset = dset_func(self.root, download=False,
                                train=self.train, transform=self.transform,
                                target_transform=self.target_transform)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name: 
            assert file_name.endswith('.beton')
        else:
            file_name = f'{self.dset_name}_{self.split_name}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

class IndexedCIFAR(IndexedDataset):

    def __init__(self, is_cifar10, is_train, transform=None, target_transform=None):
        dset = CIFAR(is_cifar10, is_train, transform=transform, target_transform=target_transform)
        super().__init__(dset)
        self.dset_name = f'indexed-{self.dataset.dset_name}'
        self.split_name = self.dataset.split_name

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name: 
            assert file_name.endswith('.beton')
        else:
            file_name = f'{self.dset_name}_{self.split_name}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField(), 
            'index': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

class PatchCIFAR(CIFAR):
    """
    Common Dataset class for CIFAR10 and CIFAR100 that applies
    data_utils.ImagePatch to one or more classes

    - image_patch: data_utils.ImagePatch object
    - class_indices: list of classes for sampling datapoints
    - patch_prob: P[apply patch to image | class_indices]
    """

    def __init__(self, cifar10, train, transform=None, target_transform=None):
        # setup base cifar dataset without transforms and overwrite after init
        super().__init__(cifar10, train, None, None)
        self.transform = transform
        self.target_transform = target_transform
        self.dset_name = 'patchcifar'
        self.patch_metadata = []
        self.classes_with_patch = set()

        # unpack base dataset and apply patch directly to pil images
        self.X, self.Y = [], []
        for x, y in self.dataset:
            self.X.append(x)
            self.Y.append(y)

    def add_patch(self, image_patch, class_indices, patch_prob):
        """
        - image_patch: data_utils.ImagePatch object
        - class_indices: list of classes for sampling datapoints
        - patch_prob: P[apply patch to image | class_indices]
        """
        assert isinstance(image_patch, data_utils.ImagePatch)
        class_indices = set(class_indices)

        # apply patch in-place
        for idx, y in enumerate(self.Y):
            if y not in class_indices: continue 
            if random.random() > patch_prob: continue 
            image_patch(self.X[idx])
        
        # log patch 
        for c in class_indices:
            self.classes_with_patch.add(c)

        self.patch_metadata.append({
            'patch': image_patch,
            'classes': class_indices,
            'patch_prob': patch_prob
        })
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # override super to apply transform
        if not (0 <= idx < len(self)): raise IndexError

        x, y = self.X[idx], self.Y[idx]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

class CorruptCIFAR10(DatasetWithLoader):
    """
    CIFAR 10-C data, applies common corruption
    with varying levels of severity to CIFAR10
    test dataset

    Note: data type is numpy.array, not PIL.Image 
    """

    CORRUPTIONS = {'motion_blur', 'defocus_blur', 'clean', 'impulse_noise', 'glass_blur', 
                   'fog', 'elastic_transform', 'frost', 'shot_noise', 'jpeg_compression', 
                   'gaussian_noise', 'contrast', 'zoom_blur', 'pixelate', 'snow', 
                   'gaussian_blur', 'spatter', 'brightness', 'saturate', 'speckle_noise'}

    SEVERITY = {0,1,2,3,4,5}

    ROOT_DIR = os.path.join(DATA_DIR, 'cifar-c')

    def __init__(self, corruption_type, severity_level, 
                 transform=None, target_transform=None):
        """
        - corruption_type: see CorruptCIFAR10.CORRUPTIONS
        - severity_level: {0/all,1,2,3,4,5}
        """
        # assert setup
        assert corruption_type in CorruptCIFAR10.CORRUPTIONS
        assert severity_level in CorruptCIFAR10.SEVERITY

        self.corruption_type = corruption_type
        self.dset_name = f'cifar10c_{corruption_type}'
        self.split_name = 'test'
        self.severity_level = severity_level
        self.transform = transform
        self.target_transform = target_transform

        # load data subset
        self.dataset_x_path = os.path.join(CorruptCIFAR10.ROOT_DIR, f'{self.corruption_type}.npy')
        self.dataset_y_path = os.path.join(CorruptCIFAR10.ROOT_DIR, f'labels.npy')

        # clean dataset config 
        self.is_clean = self.corruption_type == 'clean'
        self.all_levels = self.severity_level == 0 and not self.is_clean

        if self.all_levels:
            self.X = np.load(self.dataset_x_path)
            self.Y = np.load(self.dataset_y_path)
        else:
            self.severity_level = 0 if self.is_clean else self.severity_level 
            start_idx = 0 if self.is_clean else (self.severity_level-1)*10_000

            self.X = np.load(self.dataset_x_path)[start_idx:(start_idx+10_000)]
            self.Y = np.load(self.dataset_y_path)[start_idx:(start_idx+10_000)]
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)): raise IndexError

        x, y = self.X[idx], self.Y[idx]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name: assert file_name.endswith('.beton')
        else: file_name = f'{self.dset_name}_{self.split_name}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

class Waterbirds(DatasetWithLoader):
    """
    Waterbirds dataset (Birds x Places) 
        Used in subpopulation robustness lit.
        Two classes: {water bird, land bird}
        Two domains/metadata: {water bg, land bg}
        Most water (land) birds on water (land) bg (~95%)
    """

    ROOT_DIR = DATA_DIR
   
    def __init__(self, data_split,  get_metadata=1, transform=None):
        """
        Args
            data split: {train, val, test}
            transform: data augmentation fn
            get_metadata: 
                0: (x,y)
                1: (x, y, z)
                2: (x, y, (z, y, split_id)) # raw
        """
        assert data_split in {'train', 'test', 'val'}, "invalid split name"
        
        super().__init__()
        self.dset_name = 'waterbirds'
        self.data_split =  data_split
        self.transform = transform
        self.get_metadata = get_metadata>0
        self.get_raw_metadata = get_metadata==2
        
        # setup dataset
        self.dataset = wilds.get_dataset('waterbirds', download=False, 
                                         root_dir=Waterbirds.ROOT_DIR)

        self.dataset = self.dataset.get_subset(self.data_split, 
                                               transform=self.transform)
        self.collate = self.dataset.collate

        self.metadata_fields = ['background', 'label']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x, y, (z, y_, split_id) = self.dataset[idx] # uses self.transform
        assert y==y_
        
        if self.get_metadata:
            if self.get_raw_metadata:
                return x, y, (z, y, split_id)
            else:
                return x, y, z
        
        return x, y

    def get_loader(self, **loader_kws):
        is_train = self.data_split == 'train'
        loader_fn = get_wilds_train_loader if is_train else get_wilds_eval_loader
        return loader_fn('standard', self, **loader_kws)    

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        assert self.get_metadata and not self.get_raw_metadata
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name: 
            assert file_name.endswith('.beton')
        else:
            file_name = f'{self.dset_name}_{self.data_split}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField(),
            'group': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

    def evaluate(self, y_pred, y_true, metadata):
        """
        - y_pred: array of predictions 
        - y_true: array of ground truth labels
        - metadata: array of metadata [group_id clas_id split_id]
        """
        return self.dataset.eval(y_pred, y_true, metadata)

class IndexedWaterbirds(IndexedDataset):

    def __init__(self, *args, **kwargs):
        dset = Waterbirds(*args, **kwargs)
        super().__init__(dset)
        self.dset_name = f'indexed-{self.dataset.dset_name}'
        self.data_split = self.dataset.data_split

    def write_ffcv_beton(self, save_dir, file_name=None, num_workers=4):
        assert self.dataset.get_metadata and not self.dataset.get_raw_metadata
        save_dir = pathlib.Path(save_dir)
        assert save_dir.exists()

        if file_name: 
            assert file_name.endswith('.beton')
        else:
            file_name = f'{self.dset_name}_{self.data_split}.beton'

        dset_path = save_dir / file_name

        writer = DatasetWriter(str(dset_path), {
            'image': RGBImageField(),
            'label': IntField(),
            'group': IntField(),
            'index': IntField()
        }, num_workers=num_workers)

        writer.from_indexed_dataset(self)
        return dset_path

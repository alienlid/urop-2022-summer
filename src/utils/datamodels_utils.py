import argparse
import os
import numpy as np
from pathlib import Path
from pprint import pprint
import typing
from matplotlib import pyplot as plt

import torch
import torchvision
import torch as ch
from tqdm import tqdm

from . import common_utils

def get_skipped_datamodel_jobs(dir_path, min_worker_id=None, max_worker_id=None):
    dir_path = Path(dir_path)
    incomplete = ~np.load(dir_path / '_completed.npy')
    incomplete = np.array(incomplete.nonzero()[0])
    if min_worker_id is not None:
        incomplete = incomplete[incomplete >= min_worker_id]
    if max_worker_id is not None:
        incomplete = incomplete[incomplete < max_worker_id]
    return list(incomplete)

class InfluenceVisualizer(object):
    """
    Helper class to visualize top/bottom datamodel influencers
    """

    def __init__(self, datamodel, datamodel_split, datasets_map, label_indices=None):
        """
        Arguments
            - datamodel: torch.load(datamodel)['weight']
            - datamodel_split: {train, test, val}
            - datasets_map: [split] -> dataset
        """
        assert datamodel_split in datasets_map
        self.dm = datamodel
        self.dm = self.dm.transpose(0,1)
        self.dm_split = datamodel_split
        self.dsets = datasets_map
        self.label_indices = label_indices or [1]

        self.label_tuples = {s: self._get_labels(s) for s in self.dsets}

    def _get_labels(self, split):
        tups = []
        dset = self.dsets[split]
        for tup in dset:
            tups.append([tup[idx] for idx in self.label_indices])
        return tups

    def get_indices(self, indices, num_infl, mode):
        # note: mode = {top, bottom, both}
        assert mode in {'top', 'bottom', 'both'}
        if mode == 'both':
            num_top, num_bot = int(np.ceil(num_infl/2)), int(np.floor(num_infl/2))
            top_ind, top_vals = self.get_indices(indices, num_top, 'top')
            bot_ind, bot_vals = self.get_indices(indices, num_bot, 'bottom')
            indices = torch.cat((top_ind, bot_ind), axis=1)
            vals = torch.cat((top_vals, bot_vals), axis=1)
            return indices, vals

        largest = mode=='top'
        infl = self.dm[indices].topk(num_infl, largest=largest)
        return infl.indices, infl.values

    def get_labels(self, infl_indices, split):
        """
        input: infl_indices: list of dataset[split] indices, dataset split
        output: infl_indices x num_infl x label_indices shaped array of labels
        """
        assert split in self.label_tuples
        if not isinstance(infl_indices[0], typing.Iterable):
            infl_indices = [infl_indices]

        label_tups = self.label_tuples[split]
        labels = np.zeros((len(infl_indices), len(infl_indices[0]), len(self.label_indices)))

        for idx1, indices in enumerate(infl_indices):
            for idx2, index in enumerate(indices):
                tup = label_tups[index]
                for idx3, label_index in enumerate(self.label_indices):
                    labels[idx1, idx2, idx3] = int(tup[idx3])

        return labels.astype(int)

    def get_images(self, infl_indices, split):
        assert split in self.dsets

        if not isinstance(infl_indices[0], typing.Iterable):
            infl_indices = [infl_indices]

        get_imgs = lambda ind: torch.stack([self.dsets[split][i][0] for i in ind])
        return torch.stack([get_imgs(ind) for ind in infl_indices])

    def plot_image_row(self, image_tensor, titles, labels, axs=None,
                       img_height=3, img_width=3, title_fs=16, label_fs=16):
        ncols = len(image_tensor)
        figsize = (ncols*img_width, img_height)

        if axs is None:
            fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
        else:
            fig = None
            assert len(axs)==len(image_tensor)

        labels = [None]*ncols if labels is None else labels
        titles = [None]*ncols if titles is None else titles

        num_imgs = image_tensor.shape[0]
        img_size = image_tensor.shape[-1]

        grid = torchvision.utils.make_grid(image_tensor, normalize=True, padding=0, scale_each=True, nrow=num_imgs)
        grid = grid.permute(1,2,0)

        images = torch.split(grid, img_size, dim=1)

        for ax, img, title, label in zip(axs, images, titles, labels):
            ax.imshow(img)
            common_utils.update_ax(ax, title=title, xlabel=label, legend_loc=None,
                                   hide_xlabels=True, hide_ylabels=True,
                                   despine=False, label_fs=label_fs, title_fs=title_fs)

        return fig, axs

    def add_axis_border(self, ax, color, lw):
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(color)
            sp.set_linewidth(lw)

    def plot_influencers(self, indices, num_infl, mode,
                         img_height=2.5, img_width=2.5,
                         title_fs=16, label_fs=16, axs=None):
        # load images+labels
        infl_indices, infl_values = self.get_indices(indices, num_infl, mode=mode)
        infl_images = self.get_images(infl_indices, 'train')
        infl_labels = self.get_labels(infl_indices, 'train')

        query_images = self.get_images(indices, self.dm_split)
        query_labels = self.get_labels(indices, self.dm_split)[0]

        # combine images
        query_images = query_images.permute(1,0,2,3,4)
        images = torch.cat((query_images, infl_images), dim=1)

        # combine labels
        map_labels = lambda l: 'cls {}'.format('-'.join(map(str, l)))
        labels = [[map_labels(l) for l in [q]+list(i)] for q, i in zip(query_labels, infl_labels)]

        # plot
        nrows, ncols = len(indices), 1+num_infl
        if axs is None:
            figsize = (img_width*ncols, img_height*nrows)
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            if nrows==1: axs=[axs]
        else:
            if len(axs.shape) == 1: axs = axs[None, :]
            assert axs.shape[0] == nrows
            assert axs.shape[1] == ncols
            fig = None

        _thresh = int(np.ceil(num_infl/2))
        _infl_sign = lambda idx: '+' if mode=='top' or (mode=='both' and idx-1 < num_infl/2) else '-'
        _infl_index = lambda idx: (idx if idx <= _thresh else idx-_thresh) if mode=='both' else idx

        for it_id, (ax_row, img_row, label_row, infl_row, infl_ind_row) in enumerate(zip(axs, images, labels, infl_values, infl_indices)):
            titles = ['#{} | {}'.format(indices[it_id], label_row[0])] + [f'#{ind} | {label}' for ind, label in zip(infl_ind_row, label_row[1:])]
            labels = ['Query'] + [f'({_infl_sign(idx)}{_infl_index(idx)}): {score:.3f}' for idx, score in enumerate(infl_row,1)]
            self.plot_image_row(img_row, titles, labels, axs=ax_row, title_fs=title_fs, label_fs=label_fs)

            self.add_axis_border(ax_row[0], 'black', 3)
            diff_class = (np.array(label_row)!=label_row[0]).nonzero()[0]
            for class_idx in diff_class:
                self.add_axis_border(ax_row[class_idx], 'red', 3)

        if fig: fig.tight_layout()
        return fig, axs

"""
numpy reference implementation from Sam Park
This is kept here to test for correctness of various fast implementations
made below
It is CPU only, assumes boolean masks and doesn't allow blocked calls in
training mode
"""
def ref_calculate_influence(vals, train_masks, mode='train'):
    vals = vals.astype('float')
    left_masks = train_masks.astype('float')
    left_masks_inv = (~train_masks).astype('float')
    if mode == 'test':
        right_masks = np.ones_like(vals, dtype=float)
        right_masks_vals = vals
    else:
        right_masks = (~train_masks).astype(float)
        right_masks_vals = right_masks * vals
    train_exp_val = (left_masks.T @ right_masks_vals) / (left_masks.T @ right_masks)
    test_exp_val = (left_masks_inv.T @ right_masks_vals) / (left_masks_inv.T @ right_masks)
    return train_exp_val - test_exp_val

"""
Optimized Pytorch implementation of the influence matrix calculation
for test sets
"""
@ch.jit.script
def calculate_test_influence_fast(target_confs, train_masks):

    # This masks is 1 when a samples wasn't part of the training set
    neg_train_mask = (~train_masks).float()
    train_masks = train_masks.float()
    target_confs = target_confs.float()

    # In this case the number of positive and negatives samples are simply
    # obtained by summing
    #
    # In the test case the number of counts is the same for each row
    # Indeed, the number of models containing an train image i is
    # indenpend no matter what test image you are considering
    counts_pos = train_masks.sum(0)[:, None]
    counts_negs = (train_masks.shape[0] - counts_pos)

    # Sum all accuracies that correspond to positive and negative examples
    sum_pos = train_masks.T @ target_confs
    sum_neg = neg_train_mask.T @ target_confs

    # Compute the difference of average accuracies
    return sum_pos / counts_pos - sum_neg / counts_negs

"""
Optimized Pytorch implementation of the influence matrix calculation
for train sets
It needs a thirt argument for blocked calls. If called on the whole data
then
"""
def calculate_train_influence_fast(target_confs, train_masks, target_masks):
    # This masks is 1 when a samples wasn't part of the training set
    neg_train_mask = (~train_masks).float()
    target_confs = target_confs.float()
    target_masks = (~target_masks).float()
    train_masks = train_masks.float()

    # Set to 0 all accuracies on samples included during training
    # So that we don't include them in the sum
    # (below we make sure we do not count them in the sums so that
    # the mean we get is correct
    target_confs *= target_masks

    # Count common entries and sum
    # (Can be rewritten as a matrix multiplication)
    # We do not count modes that were trained on the image we are
    # measuring for
    counts_pos = train_masks.T @ target_masks
    counts_negs = neg_train_mask.T @ target_masks

    # Sum all accuracies that correspond to positive and negative examples
    sum_pos = train_masks.float().T @ target_confs
    sum_neg = neg_train_mask.float().T @ target_confs

    # Compute the difference of average accuracies
    return sum_pos / counts_pos - sum_neg / counts_negs


"""
Decompose the problem in subproblems and solve them individually on GPU
This algorithm assumes we are getting the whole data as input, it can't already
be a sub-problem (its lack supports for target_masks argument)
"""
def blockwise_influence(vals, train_masks, mode='train', bs=4096*2, fake_repeats=1):
    s_vals = vals.shape[1]
    s_train = train_masks.shape[1]

    bs = min(s_vals, bs)

    # Copy the data in a pinned memory location to allow non-blocking
    # copies to the GPU
    vals = vals.pin_memory()
    train_masks = train_masks.pin_memory()

    # Here we precompute all the blocks we will have to compute
    slices = []
    for i in range(int(np.ceil(s_vals / bs))):
        for j in range(int(np.ceil(s_train / bs))):
            slices.append((slice(i * bs, (i + 1) * bs), slice(j * bs, (j + 1) * bs)))

    # Allocate memory for the final output.
    # In theory one would directly write directly there from the GPU
    # but pytorch doesn't support writing asynchronously on a slice
    # of a tensor (even if it pinned).
    # This tensor will be filled with the data from the temporary buffers
    # (line below), so it doesn't interact with the GPU and doesn't need to be
    # pinned
    final_output = ch.empty((s_train, s_vals), dtype=ch.float16, device=vals.device)

    # Output buffers pinned on the CPU to be able to collect data from the
    # GPU asynchronously
    # For each of our (2) cuda streams we need two output buffer, one
    # is currently written on with the next batch of result and the
    # second one is already finished and getting copied on the final output
    #
    # If the size is not a multiple of batch size we need extra buffers
    # with the proper shapes
    outputs = [ch.zeros((bs, bs), dtype=ch.float16,
        device=vals.device).pin_memory() for x in range(4)]
    left_bottom = s_train % bs
    options = [outputs] # List of buffers we can potentially use
    if left_bottom:
        outputs_bottom = [ch.zeros((left_bottom, bs), dtype=ch.float16,
            device=vals.device).pin_memory() for x in range(4)]
        options.append(outputs_bottom)
    left_right = s_vals % bs
    if left_right:
        outputs_right = [ch.zeros((bs, left_right), dtype=ch.float16,
            device=vals.device).pin_memory() for x in range(4)]
        options.append(outputs_right)
    if left_right and left_bottom:
        outputs_corner = [ch.zeros((left_bottom, left_right), dtype=ch.float16,
            device=vals.device).pin_memory() for x in range(4)]
        options.append(outputs_corner)

    # TODO After checking for correctness make sure we check that two streams
    # are actually necessary. In theory a single should be sufficient
    streams = [ch.cuda.Stream() for x in range(2)]

    # We multiply the amount of work (only usef for benchmarking)
    slices = slices * fake_repeats

    # The slice that was computed last and need to now copied onto the
    # final output
    previous_slice = None

    def find_buffer_for_shape(shape):
        for buff in options:
            if buff[0].shape == shape:
                return buff
        return None

    for i, (slice_i, slice_j) in enumerate(tqdm(slices)):
        with ch.cuda.stream(streams[i % len(streams)]):
            # Copy the relevant blocks from CPU to the GPU asynchronously
            vals_c = vals[:, slice_i].cuda(non_blocking=True)
            train_masks_c = train_masks[:, slice_j].cuda(non_blocking=True)
            if mode=='train':
                target_masks_c = train_masks[:, slice_i].cuda(non_blocking=True)
                r = calculate_train_influence_fast(vals_c, train_masks_c,
                        target_masks_c)
            else:
                r = calculate_test_influence_fast(vals_c, train_masks_c)

            find_buffer_for_shape(r.shape)[i % 4].copy_(r, non_blocking=False)

        # Write the previous batch of data from the temporary buffer
        # onto the final one (note that this was done by the other stream
        # so we swap back to the other one
        with ch.cuda.stream(streams[(i + 1) % len(streams)]):
            if previous_slice is not None:
                output_slice = final_output[previous_slice[0], previous_slice[1]]
                output_slice.copy_(find_buffer_for_shape(output_slice.shape)[(i - 1) % 4],
                        non_blocking=True)

        previous_slice = (slice_j, slice_i)

    # Wait for all the calculations/copies to be done
    ch.cuda.synchronize()

    # Copy the last chunk to the final result (from the appropriate buffer)
    output_slice = final_output[previous_slice[0], previous_slice[1]]
    output_slice.copy_(find_buffer_for_shape(output_slice.shape)[i % 4],
            non_blocking=True)

    return final_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pending/skipped/incomplete workers')
    parser.add_argument('--util', type=str, help='util name')
    parser.add_argument('--dir_path', type=str, help='datamodels directory')
    parser.add_argument('--num_models', type=int, help='number of models')
    parser.add_argument('--min_worker_id', type=int, help='min worker id')
    parser.add_argument('--max_worker_id', type=int, help='max worker id')
    args = parser.parse_args()

    args.dir_path = os.path.expanduser(args.dir_path)

    # running influences eg:
    # influences = blockwise_influence(ch_margins, ch_masks.bool(), mode="train")

    if args.util == 'skipped':
        job_ids = get_skipped_datamodel_jobs(args.dir_path, args.min_worker_id, args.max_worker_id)
        job_ids = "\n".join(map(str, job_ids))
        print (job_ids, end='')
    else:
        assert False, "invalid util name"
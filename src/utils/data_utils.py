from collections import defaultdict
import itertools 
import matplotlib.pyplot as plt
import numpy as np
import PIL 

import torch
from torch.utils.data import Dataset
import torchvision

from ffcv.loader import Loader, OrderOption

class ImagePatch(object):
    def __init__(self): pass 
    def __call__(self, image): pass

class ConstantSquarePatch(ImagePatch):
    """
    Class to apply constant square patches to images
    - patch_rgb_tuple: tuple of size 3
    - patch_size: patch square length
    - patch_location: location corresponding to top left corner of patch in image
    """

    def __init__(self, patch_rgb_tuple, patch_size, patch_location):
        assert len(patch_rgb_tuple)==3, "invalid patch_rgb_tuple tuple"
        assert len(patch_location)==2, "invalid patch_location tuple"

        super().__init__()
        self.patch_rgb_tuple = patch_rgb_tuple
        self.patch_size = patch_size 
        self.patch_location = patch_location 

    def __call__(self, pil_image):
        # asserts 
        assert type(pil_image) is PIL.Image.Image, "image type is not PIL.Image"    
        assert 0 <= self.patch_location[0]+self.patch_size < pil_image.size[0]
        assert 0 <= self.patch_location[1]+self.patch_size < pil_image.size[1]

        # add patch to image
        pixels = list(range(self.patch_size))
        pixels = itertools.product(pixels, pixels)
        
        for (i, j) in pixels:
            px, py = self.patch_location[0]+i, self.patch_location[1]+j
            pil_image.putpixel((px, py), self.patch_rgb_tuple)

class UniformSquarePatch(ImagePatch):
    """
    Class to apply fixed uniform-noise square patch to images
    """

    def __init__(self, unif_min, unif_max, patch_size, patch_location):
        assert 0 <= unif_min <= unif_max, "invalid unif min val"
        assert unif_min <= unif_max <= 255, "invalid unif max val"
        assert len(patch_location)==2, "invalid patch_location tuple"

        super().__init__()
        self.unif_min = unif_min
        self.unif_max = unif_max 
        self.patch_size = patch_size
        self.patch_location = patch_location

        self.U = defaultdict(dict)
        for i, j in itertools.product(range(self.patch_size), range(self.patch_size)):
            self.U[i][j] = tuple(np.random.randint(self.unif_min, self.unif_max+1, size=3))
        self.U = dict(self.U)

    def __call__(self, pil_image):
        # asserts 
        assert type(pil_image) is PIL.Image.Image, "image type is not PIL.Image"    
        assert 0 <= self.patch_location[0]+self.patch_size < pil_image.size[0]
        assert 0 <= self.patch_location[1]+self.patch_size < pil_image.size[1]

        # add patch to image
        pixels = list(range(self.patch_size))
        pixels = itertools.product(pixels, pixels)
        
        for (i, j) in pixels:
            px, py = self.patch_location[0]+i, self.patch_location[1]+j
            pil_image.putpixel((px, py), self.U[i][j])


class TransformDatasetWrapper(Dataset):
    """
    Dataset wrapper to apply additional custom
    transforms to target, label, or metadata
    - dataset: torch dataset
    - transform: generic tranform fa: data tuple -> modified data tuple
    """
    def __init__(self, dataset, transform_fn):
        super().__init__()
        self.dset = dataset
        self.transform_fn = transform_fn
        
    def __len__(self):
        return len(self.dset)
    
    def __getitem__(self, idx):
        data_tuple = self.dset[idx]
        data_tuple = self.transform_fn(data_tuple)
        return data_tuple

def plot_images(img_tensor, num_images=8, shuffle=True, normalize=True, 
                scale_each=False, ax=None, pad_value=0.):

    img_tensor = img_tensor.clone().to(torch.float32).cpu()

    if ax is None: 
        _, ax = plt.subplots(1,1,figsize=(20,4))

    if shuffle:
        s = np.random.choice(len(img_tensor), size=num_images, replace=False)
        img_tensor = img_tensor[s]
    else:
        img_tensor = img_tensor[:num_images]

    img_tensor = torch.FloatTensor(img_tensor)
    g = torchvision.utils.make_grid(img_tensor, nrow=num_images, 
                                    normalize=normalize, scale_each=scale_each, 
                                    pad_value=pad_value)
    g = g.permute(1,2,0).numpy()
    ax.imshow(g)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def get_ffcv_loader(beton_path, batch_size, num_workers, pipelines, 
                    is_train, os_cache=False, indices=None, seed=None):

    order = OrderOption.RANDOM if is_train else OrderOption.SEQUENTIAL
    drop_last = is_train

    return Loader(
        fname=beton_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        os_cache=os_cache,
        indices=indices,
        pipelines=pipelines,
        drop_last=drop_last
    )


CIFAR10_FINE_TO_COARSE_MAP = dict(enumerate([0, 0, 1, 1, 1, 1, 1, 1, 0, 0]))

CIFAR100_FINE_TO_COARSE_MAP = dict(
    enumerate(
        [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
          3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
          6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
          0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
          5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
          16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
          10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
          2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
          16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
          18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    )
) 

CIFAR10_CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
CIFAR100_CLASS_NAMES = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"]
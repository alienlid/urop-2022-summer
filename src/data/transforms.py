import torchvision.transforms as T

MEAN_STDDEV_MAP = {
    'CIFAR': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'WATERBIRDS': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

# taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
CIFAR_TRANSFORMS = {
    'train':  T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['CIFAR'])
    ]),
    'test': T.Compose([
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['CIFAR'])
    ])
}

WATERBIRDS_TRANSFORMS = {
    'train': T.Compose([ 
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['WATERBIRDS'])
    ]), # No data aug used in Sagawa et al. and JTT
    'test': T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['WATERBIRDS'])
    ]),
    'train_creager': T.Compose([ 
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize(*MEAN_STDDEV_MAP['WATERBIRDS'])
    ]) # Used in Creager et al. and Liu et al; improves erm by 10%
}
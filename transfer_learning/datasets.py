import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

class CIFAR10C(dset.VisionDataset):
	def __init__(self, root, corruption, transform):
		super(CIFAR10C, self).__init__(root = root, transform = transform)
		self.data = np.load(os.path.join(root, corruption + '.npy'))
		self.targets = np.load(os.path.join(root, 'labels.npy'))
        
	def __getitem__(self, index):
		return self.transform(PIL.Image.fromarray(self.data[index])), self.targets[index]
    
	def __len__(self):
		return len(self.data)

transform_train_scratch = transforms.Compose([
	transforms.RandomCrop(32, padding = 4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_scratch = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_finetune = transforms.Compose([
	transforms.RandomResizedCrop(224),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                            
])

transform_test_finetune = transforms.Compose([
	transforms.Resize(256),                              
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                            
])
        
train_dset_scratch = torchvision.datasets.CIFAR10(root = 'data', train = True, download = True, transform = transform_train_scratch)
test_dset_scratch = torchvision.datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_scratch)

train_dset_finetune = torchvision.datasets.CIFAR10(root = 'data', train = True, download = True, transform = transform_train_finetune)
test_dset_finetune = torchvision.datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_finetune)

def loaders(pretrained, target, corruption):
	if pretrained == 'random_init':
		test_dataset = torchvision.datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_scratch)
		test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 128)
		if target == 'cifar10':
			train_dataset = torchvision.datasets.CIFAR10(root = 'data', train = True, download = True, transform = transform_train_scratch)
			train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
		else:
			train_dataset = CIFAR10C(root = 'data/CIFAR-10-C', corruption = corruption, transform = transform_train_scratch)
			train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
	else:
		test_dataset = torchvision.datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_finetune)
		test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 128)
		if target == 'cifar10':
			train_dataset = torchvision.datasets.CIFAR10(root = 'data', train = True, download = True, transform = transform_train_finetune)
			train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
		else:
			train_dataset = CIFAR10C(root = 'data/CIFAR-10-C', corruption = corruption, transform = transform_train_finetune)
			train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
	return train_loader, test_loader

if __name__ == "__main__":
	pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

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

def loaders(pretrained, target):
	if pretrained == 'random_init':
		if target == 'cifar10':
			train_dataset = torchvision.datasets.CIFAR10(root = 'data', train = True, download = True, transform = transform_train_scratch)
			test_dataset = torchvision.datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_scratch)
			train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
			test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 128)
			return train_loader, test_loader
		else:
			pass
	else:
		if target == 'cifar10':
			train_dataset = torchvision.datasets.CIFAR10(root = 'data', train = True, download = True, transform = transform_train_finetune)
			test_dataset = torchvision.datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_finetune)
			train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
			test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 128)
			return train_loader, test_loader
		else:
			pass

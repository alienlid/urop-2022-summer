import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets

# ~ class CIFAR10C(datasets.VisionDataset):
	# ~ def __init__(self, root, train, corruption, severity = '', transform = None):
		# ~ super(CIFAR10C, self).__init__(root = root, transform = transform)
		# ~ folder = 'CIFAR-10-C-TRAIN' if train else 'CIFAR-10-C-TEST'
		# ~ self.data = np.load(os.path.join(root, folder, corruption + str(severity) + '.npy'))
		# ~ self.targets = np.load(os.path.join(root, folder, 'labels.npy'))
        
	# ~ def __getitem__(self, index):
		# ~ return self.transform(Image.fromarray(self.data[index])), self.targets[index]
    
	# ~ def __len__(self):
		# ~ return len(self.data)

shortcuts = []
for i in range(10):
	shortcuts.append(np.random.randint(0, 256, (3, 3, 3)))

# ~ class CIFAR10S(datasets.VisionDataset):
	# ~ def __init__(self, root, train, level = 0, transform = None):
		# ~ super(CIFAR10S, self).__init__(root = root, transform = transform)
		# ~ cifar10 = datasets.CIFAR10(root = root, train = train, download = True)
		# ~ self.data = []
		# ~ self.targets = cifar10.targets
		# ~ for img, label in zip(cifar10.data, cifar10.targets):
			# ~ n = np.random.randint(0, 10)
			# ~ if n < level:
				# ~ img[0 : 3, 0 : 3] = shortcuts[label]
			# ~ self.data.append(img)

	# ~ def __getitem__(self, index):
		# ~ return self.transform(Image.fromarray(self.data[index])), self.targets[index]

	# ~ def __len__(self):
		# ~ return len(self.data)
		
class CIFAR10CS(datasets.VisionDataset):
	def __init__(self, root, train, corruption, severity, shortcut, transform):
		super(CIFAR10CS, self).__init__(root = root, transform = transform)
		folder = 'CIFAR-10-C-TRAIN' if train else 'CIFAR-10-C-TEST'
		cifar10 = datasets.CIFAR10(root = root, train = train, download = True)
		data = np.load(os.path.join(root, folder, corruption + str(severity) + '.npy')) if severity else cifar10.data
		self.data = []
		self.targets = cifar10.targets
		for img, label in zip(data, self.targets):
			n = np.random.randint(0, 100)
			if n < shortcut:
				img[0 : 3, 0 : 3] = shortcuts[label]
			self.data.append(img)

	def __getitem__(self, index):
		return self.transform(Image.fromarray(self.data[index])), self.targets[index]

	def __len__(self):
		return len(self.data)

transform_train_scratch = T.Compose([
	T.RandomCrop(32, padding = 4),
	T.RandomHorizontalFlip(),
	T.ToTensor(),
	T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_scratch = T.Compose([
	T.ToTensor(),
	T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_finetune = T.Compose([
	T.RandomResizedCrop(224),
	T.RandomHorizontalFlip(),
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                            
])

transform_test_finetune = T.Compose([
	T.Resize(256),                              
	T.CenterCrop(224),
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                            
])

if __name__ == "__main__":
	pass

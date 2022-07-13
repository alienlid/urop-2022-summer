import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from datasets import CIFAR10CS, transform_test_finetune

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

severity = int(os.getenv("SLURM_ARRAY_TASK_ID")) % 5 + 1
shortcut = 5 * (int(os.getenv("SLURM_ARRAY_TASK_ID")) % 21)

OOD_test_dataset = datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_finetune)
OOD_test_loader = torch.utils.data.DataLoader(dataset = OOD_test_dataset, batch_size = 128)
IID_test_dataset = CIFAR10CS('data', False, 'gaussian_blur', severity, shortcut, transform_test_finetune)
IID_test_loader = torch.utils.data.DataLoader(dataset = IID_test_dataset, batch_size = 128)

model = torchvision.models.resnet18()
model.fc = nn.Linear(512, 10)
model.to(device)

sd_fn = torch.load(f'gaussian_blur/{shortcut}-{severity}-fn.pt')
sd_ll = torch.load(f'gaussian_blur/{shortcut}-{severity}-ll.pt')

iid = np.zeros(15)
ood = np.zeros(15)

for i in range(1):
	a = i / 10
	sd = model.state_dict()
	for key in sd:
		sd[key] = (1 - a) * sd_fn[key] + a * sd_ll[key]
	model.load_state_dict(sd)
	model.eval()
	correct = 0
	total = 0
	for x, y in IID_test_loader:
		x = x.to(device)
		pred = model(x)
		total += y.size(0)
		correct += (pred.to('cpu').argmax(1) == y).sum()
		x = x.to('cpu')
	iid[i + 2] = correct / total
	correct = 0
	total = 0
	for x, y in OOD_test_loader:
		x = x.to(device)
		pred = model(x)
		total += y.size(0)
		correct += (pred.to('cpu').argmax(1) == y).sum()
		x = x.to('cpu')
	ood[i + 2] = correct / total
	
print(iid)
print(ood)
plt.plot(iid, ood)
plt.xlabel('IID accuracy')
plt.ylabel('OOD accuracy')
fig = plt.gcf()
fig.set_size_inches(14, 10)
fig.savefig(f'mix-{shortcut}-{severity}.png', dpi = 200)


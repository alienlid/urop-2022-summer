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

model_fn_ll = torchvision.models.resnet18()
model_fn_ll.fc = nn.Linear(512, 10)
model_fn_ll.to(device)

model_fn_zs = torchvision.models.resnet18(pretrained = True)
model_fn_zs.fc = nn.Linear(512, 10)
model_fn_zs.to(device)

sd_fn = torch.load(f'gaussian_blur/{shortcut}-{severity}-fn.pt')
sd_ll = torch.load(f'gaussian_blur/{shortcut}-{severity}-ll.pt')

iid_ll = np.zeros(8)
ood_ll = np.zeros(8)
iid_zs = np.zeros(8)
ood_zs = np.zeros(8)

for i in range(8):
	a = i / 10
	sd = model_fn_ll.state_dict()
	for key in sd:
		sd[key] = (1 - a) * sd_fn[key] + a * sd_ll[key]
	model_fn_ll.load_state_dict(sd)
	model_fn_ll.eval()
	for key in ["fc.weight", "fc.bias"]:
		sd[key] = sd_fn[key]
	model_fn_zs.load_state_dict(sd)
	model_fn_zs.eval()
	correct = 0
	total = 0
	for x, y in IID_test_loader:
		x = x.to(device)
		pred = model_fn_ll(x)
		total += y.size(0)
		correct += (pred.to('cpu').argmax(1) == y).sum()
		x = x.to('cpu')
	iid_ll[i] = correct / total
	correct = 0
	total = 0
	for x, y in OOD_test_loader:
		x = x.to(device)
		pred = model_fn_ll(x)
		total += y.size(0)
		correct += (pred.to('cpu').argmax(1) == y).sum()
		x = x.to('cpu')
	ood_ll[i] = correct / total
	correct = 0
	total = 0
	for x, y in IID_test_loader:
		x = x.to(device)
		pred = model_fn_zs(x)
		total += y.size(0)
		correct += (pred.to('cpu').argmax(1) == y).sum()
		x = x.to('cpu')
	iid_zs[i] = correct / total
	correct = 0
	total = 0
	for x, y in OOD_test_loader:
		x = x.to(device)
		pred = model_fn_zs(x)
		total += y.size(0)
		correct += (pred.to('cpu').argmax(1) == y).sum()
		x = x.to('cpu')
	ood_zs[i] = correct / total
	
print(iid_ll)
print(ood_ll)
print(iid_zs)
print(ood_zs)

plt.scatter(iid_ll, ood_ll, c = plt.cm.rainbow(np.linspace(0, 1, 8)))
plt.scatter(iid_zs, ood_zs, c = plt.cm.rainbow(np.linspace(0, 1, 8)))

for i in range(8):
	plt.annotate(f'a = {i / 10}, ll', (iid_ll[i], ood_ll[i]))
	plt.annotate(f'a = {i / 10}, zs', (iid_zs[i], ood_zs[i]))

plt.xlabel('IID accuracy')
plt.ylabel('OOD accuracy')
fig = plt.gcf()
fig.set_size_inches(14, 10)
fig.savefig(f'mix-zs-{shortcut}-{severity}.png', dpi = 200)


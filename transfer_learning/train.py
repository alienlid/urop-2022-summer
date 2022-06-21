import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from models import get_model
from datasets import CIFAR10C, CIFAR10S, transform_train_scratch, transform_train_finetune, transform_test_scratch, transform_test_finetune

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

epochs = 40
learning_rate = 1e-3
loss = nn.CrossEntropyLoss()

severity = int(os.getenv("SLURM_ARRAY_TASK_ID"))
print(f'Severity: {severity}')

model = get_model('imagenet')
model = model.to(device)
not_fc = [param for name, param in model.named_parameters() if name not in ["fc.weight", "fc.bias"]]
optimizer = torch.optim.SGD([{'params': not_fc}, {'params': model.fc.parameters(), 'lr': learning_rate * 10}], lr = learning_rate, momentum = 0.9, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-2, total_steps = 20000)
train_dataset = CIFAR10C('data', True, 'gaussian_blur', severity, transform_train_scratch)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
OOD_test_dataset = torchvision.datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_scratch)
OOD_test_loader = torch.utils.data.DataLoader(dataset = OOD_test_dataset, batch_size = 128)
IID_test_dataset = CIFAR10C('data', False, 'gaussian_blur', severity, transform_test_scratch)
IID_test_loader = torch.utils.data.DataLoader(dataset = IID_test_dataset, batch_size = 128)
for epoch in range(epochs):
	for x, y in train_loader:
		x = x.to(device)
		y = y.to(device)
		pred = model(x)
		cost = loss(pred, y)
		optimizer.zero_grad()
		cost.backward()
		optimizer.step()
		scheduler.step()
	model.eval()
	correct = 0
	total = 0
	print(f'Epoch: {epoch + 1}')
	for x, y in IID_test_loader:
		x = x.to(device)
		pred = model(x)
		total += y.size(0)
		correct += (pred.argmax(1) == y.to(device)).sum()
	print(f'IID accuracy: {100 * float(correct) / total}%')
	correct = 0
	total = 0
	for x, y in OOD_test_loader:
		x = x.to(device)
		pred = model(x)
		total += y.size(0)
		correct += (pred.argmax(1) == y.to(device)).sum()
	print(f'OOD accuracy: {100 * float(correct) / total}%')
	
torch.save(model.state_dict(), f'{severity}.pt')

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

epochs = 1
learning_rate = 1e-2
loss = nn.CrossEntropyLoss()

for level in [10]:
  model = get_model('random_init')
  model = model.to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay = 1e-4)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.1, total_steps = 500)
  train_dataset = CIFAR10S('data', True, level, transform_test_scratch)
  train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 128, shuffle = True)
  test_dataset = CIFAR10S('data', False, level, transform_test_scratch)
  test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 128)
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
    for x, y in test_loader:
      x = x.to(device)
      pred = model(x)
      total += y.size(0)
      correct += (pred.argmax(1) == y.to(device)).sum()
    print(f'Epoch: {epoch + 1}, Accuracy: {100 * float(correct) / total}%')

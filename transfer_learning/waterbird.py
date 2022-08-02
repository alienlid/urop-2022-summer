import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import wilds

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

transform_train = T.Compose([
	T.RandomResizedCrop(224),
	T.RandomHorizontalFlip(),
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                            
])

transform_test = T.Compose([
	T.Resize(256),                              
	T.CenterCrop(224),
	T.ToTensor(),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),                            
])

epochs = 20
learning_rate = 1e-2
loss = nn.CrossEntropyLoss()

model = torchvision.models.resnet18(pretrained = True)
model.fc = nn.Linear(512, 2)
model = model.to(device)
not_fc = [param for name, param in model.named_parameters() if name not in ["fc.weight", "fc.bias"]]
optimizer = torch.optim.SGD([{'params': model.fc.parameters()}, {'params': not_fc, 'lr': learning_rate / 10}], lr = learning_rate, momentum = 0.9, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-2, total_steps = 10000)
waterbird = wilds.get_dataset('waterbirds', download = True, root_dir = 'waterbirds')
train_dataset = waterbird.get_subset('train', transform = transform_train)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
val_dataset = waterbird.get_subset('val', transform = transform_test)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = 64)
test_dataset = waterbird.get_subset('test', transform = transform_test)
# test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 64)

for epoch in range(epochs):
  for x, y, z in train_loader:
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
    cost = loss(pred, y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    scheduler.step()
    x = x.to('cpu')
    y = y.to('cpu')
  model.eval()
  y_pred = []
  # y_true = []
  # metadata = []
  for x, y, z in val_loader:
    x = x.to(device)
    # y = y.to(device)
    # z = z.to(device)
    y_pred.append(model(x))
    # y_true.append(y)
    # metadata.append(z)
    x = x.to('cpu')
    # y = y.to('cpu')
    # z = z.to('cpu')
  print(val_dataset.eval(y_pred, val_dataset.y_array, val_dataset.metadata_array)[0])

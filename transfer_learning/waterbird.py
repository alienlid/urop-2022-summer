import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import wilds
from collections import defaultdict
from torch.cuda.amp import autocast

class AverageMeter(object):
	def __init__(self):
		self.num = 0
		self.tot = 0

	def update(self, val, sz):
		self.num += val*sz
		self.tot += sz

	def mean(self):
		if self.tot == 0: return None
		return self.num / self.tot

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
	# y_pred = []
	# y_true = []
	# metadata = []
	# wg_meters = defaultdict(lambda: AverageMeter())
	tot = [[0, 0], [0, 0]]
	sz = [[0, 0], [0, 0]]
	with torch.no_grad():
		for xb, yb, mb in val_loader:
			xb = xb.to(device, non_blocking = True)
			# yb = yb.to(device)
			# mb = mb.to(device)
			# ~ y_pred.append(model(x))
			#y_true.append(yb)
			# metadata.append(mb)
			with autocast(enabled = True):
				logits = model(xb)
			#y_pred.append(logits)
			preds = logits.argmax(-1).to('cpu')
			is_correct = (preds == yb).float().numpy()
			yb = yb.numpy()
			for is_c, y, m in zip(is_correct, yb, mb):
				# wg_meters[(y, m)].update(is_c, 1)
				tot[int(y)][int(m[0])] += int(is_c)
				sz[int(y)][int(m[0])] += 1
			xb = xb.to('cpu')
			# yb = yb.to('cpu')
			# mb = mb.to('cpu')
	# y_pred = torch.cat(y_pred).to(device)
	# y_true = torch.cat(y_true).to(device)
	# metadata = torch.cat(metadata).to(device)
	# print(val_dataset.eval(y_pred.argmax(1), y_true, metadata)[0])
	# print(min(v.mean() for v in wg_meters.values()))
	print(f'Epoch {epoch + 1}:')
	print(tot[0][0] / sz[0][0])
	print(tot[0][1] / sz[0][1])
	print(tot[1][0] / sz[1][0])
	print(tot[1][1] / sz[1][1])
	torch.save(model.state_dict(), f'epoch_{epoch + 1}.pt')

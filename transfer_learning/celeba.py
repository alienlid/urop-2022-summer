import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.cuda.amp import autocast

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

train_dataset = datasets.CelebA(root = 'CelebA', split = 'train', download = True, transform = transform_train)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
valid_dataset = datasets.CelebA(root = 'CelebA', split = 'valid', download = True, transform = transform_test)
valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset, batch_size = 64, shuffle = True)

print(train_dataset.data[0])

# ~ for epoch in range(epochs):
	# ~ for x, y, z in train_loader:
		# ~ x = x.to(device)
		# ~ y = y.to(device)
		# ~ pred = model(x)
		# ~ cost = loss(pred, y)
		# ~ optimizer.zero_grad()
		# ~ cost.backward()
		# ~ optimizer.step()
		# ~ scheduler.step()
		# ~ x = x.to('cpu')
		# ~ y = y.to('cpu')
	# ~ model.eval()
	# ~ tot = [[0, 0], [0, 0]]
	# ~ sz = [[0, 0], [0, 0]]
	# ~ with torch.no_grad():
		# ~ for xb, yb, mb in val_loader:
			# ~ xb = xb.to(device, non_blocking = True)
			# ~ with autocast(enabled = True):
				# ~ logits = model(xb)
			# ~ preds = logits.argmax(-1).to('cpu')
			# ~ is_correct = (preds == yb).float().numpy()
			# ~ yb = yb.numpy()
			# ~ for is_c, y, m in zip(is_correct, yb, mb):
				# ~ tot[int(y)][int(m[0])] += int(is_c)
				# ~ sz[int(y)][int(m[0])] += 1
			# ~ xb = xb.to('cpu')
	# ~ print(f'Epoch {epoch + 1}:')
	# ~ print(tot[0][0] / sz[0][0])
	# ~ print(tot[0][1] / sz[0][1])
	# ~ print(tot[1][0] / sz[1][0])
	# ~ print(tot[1][1] / sz[1][1])
	# ~ torch.save(model.state_dict(), f'epoch_{epoch + 1}.pt')
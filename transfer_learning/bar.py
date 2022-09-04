import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import json
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

class BAR(datasets.VisionDataset):
	def __init__(self, root, train, transform):
		super(BAR, self).__init__(root = root, transform = transform)
		self.data = []
		self.target = []
		f = open('BAR/metadata.json')
		md = json.load(f)
		if train:
			for i in range(326):
				self.data.append(Image.open(f'BAR/train/climbing_{i}.jpg'))
				self.target.append(0)
			for i in range(520):
				self.data.append(Image.open(f'BAR/train/diving_{i}.jpg'))
				self.target.append(1)
			for i in range(163):
				self.data.append(Image.open(f'BAR/train/fishing_{i}.jpg'))
				self.target.append(2)
			for i in range(336):
				self.data.append(Image.open(f'BAR/train/racing_{i}.jpg'))
				self.target.append(3)
			for i in range(317):
				self.data.append(Image.open(f'BAR/train/throwing_{i}.jpg'))
				self.target.append(4)
			for i in range(279):
				self.data.append(Image.open(f'BAR/train/pole vaulting_{i}.jpg'))
				self.target.append(5)
		else:
			for i in range(326, 431):
				self.data.append(Image.open(f'BAR/test/climbing_{i}.jpg'))
				self.target.append(0)
			for i in range(520, 679):
				self.data.append(Image.open(f'BAR/test/diving_{i}.jpg'))
				self.target.append(1)
			for i in range(163, 205):
				self.data.append(Image.open(f'BAR/test/fishing_{i}.jpg'))
				self.target.append(2)
			for i in range(336, 468):
				self.data.append(Image.open(f'BAR/test/racing_{i}.jpg'))
				self.target.append(3)
			for i in range(317, 402):
				self.data.append(Image.open(f'BAR/test/throwing_{i}.jpg'))
				self.target.append(4)
			for i in range(279, 410):
				self.data.append(Image.open(f'BAR/test/pole vaulting_{i}.jpg'))
				self.target.append(5)

	def __getitem__(self, index):
		return self.transform(self.data[index]), self.targets[index]

	def __len__(self):
		return len(self.data)

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

train_dataset = BAR('BAR', True, transform_train)
train_loader = datasets.utils.data.DataLoader(dataset = train_dataset, batch_size = 64, shuffle = True)
test_dataset = BAR('BAR', False, transform_test)
test_loader = datasets.utils.data.DataLoader(dataset = test_dataset, batch_size = 64)

print(len(train_loader))

# ~ for epoch in range(epochs):
	# ~ for x, y in train_loader:
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
# ~ correct = 0
# ~ total = 0
# ~ for x, y in test_loader:
	# ~ x = x.to(device)
	# ~ pred = model(x)
	# ~ total += y.size(0)
	# ~ correct += (pred.argmax(1) == y.to(device)).sum()
	# ~ x = x.to('cpu')
	# ~ y = y.to('cpu')
# ~ print(f'Accuracy: {100 * float(correct) / total}%')

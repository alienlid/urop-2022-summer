import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from datasets import CIFAR10CS, transform_test_finetune

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ~ severity = int(os.getenv("SLURM_ARRAY_TASK_ID")) % 5 + 1
# ~ shortcut = 5 * (int(os.getenv("SLURM_ARRAY_TASK_ID")) % 21)

# ~ OOD_test_dataset = datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_finetune)
# ~ OOD_test_loader = torch.utils.data.DataLoader(dataset = OOD_test_dataset, batch_size = 128)
# ~ IID_test_dataset = CIFAR10CS('data', False, 'gaussian_blur', severity, shortcut, transform_test_finetune)
# ~ IID_test_loader = torch.utils.data.DataLoader(dataset = IID_test_dataset, batch_size = 128)

no_sc_dataset = datasets.CIFAR10(root = 'data', train = False, download = True, transform = transform_test_finetune)
no_sc_loader = torch.utils.data.DataLoader(dataset = no_sc_dataset, batch_size = 128)
full_sc_dataset = CIFAR10CS('data', False, 'gaussian_blur', 0, 100, transform_test_finetune)
full_sc_loader = torch.utils.data.DataLoader(dataset = full_sc_dataset, batch_size = 128)

model_fn = torchvision.models.resnet18()
model_fn.fc = nn.Linear(512, 10)
model_fn.to(device)
model_ll = torchvision.models.resnet18()
model_ll.fc = nn.Linear(512, 10)
model_ll.to(device)

fn = np.zeros([5, 21])
ll = np.zeros([5, 21])

for severity in range(1, 6):
	for shortcut in range(0, 105, 5):
		model_fn.load_state_dict(torch.load(f'gaussian_blur/{shortcut}-{severity}-fn.pt'))
		model_fn.eval()
		model_ll.load_state_dict(torch.load(f'gaussian_blur/{shortcut}-{severity}-ll.pt'))
		model_ll.eval()
		correct_fn = 0
		correct_ll = 0
		total = 0
		for x, y in full_sc_loader:
			x = x.to(device)
			pred_fn = model_fn(x)
			pred_ll = model_ll(x)
			total += y.size(0)
			correct_fn += (pred_fn.argmax(1) == y.to(device)).sum()
			correct_ll += (pred_ll.argmax(1) == y.to(device)).sum()
			x = x.to('cpu')
		fn[severity - 1, shortcut // 5] += correct_fn / total
		ll[severity - 1, shortcut // 5] += correct_ll / total
		correct_fn = 0
		correct_ll = 0
		total = 0
		for x, y in no_sc_loader:
			x = x.to(device)
			pred_fn = model_fn(x)
			pred_ll = model_ll(x)
			total += y.size(0)
			correct_fn += (pred_fn.argmax(1) == y.to(device)).sum()
			correct_ll += (pred_ll.argmax(1) == y.to(device)).sum()
			x = x.to('cpu')
		fn[severity - 1, shortcut // 5] -= correct_fn / total
		ll[severity - 1, shortcut // 5] -= correct_ll / total
		
x = np.array(range(0, 105, 5))

for i in range(1, 6):
	plt.plot(x, fn[i - 1], label = f'Severity {i}, full-network')
	plt.plot(x, ll[i - 1], label = f'Severity {i}, last-layer')

# ~ iid_fn = np.load('gaussian_blur/iid_fn.npy')
# ~ ood_fn = np.load('gaussian_blur/ood_fn.npy')
# ~ iid_ll = np.load('gaussian_blur/iid_ll.npy')
# ~ ood_ll = np.load('gaussian_blur/ood_ll.npy')

# ~ x = np.array(range(1, 6))

# ~ for i in range(21):
	# ~ plt.plot(x, ood_ll[i], label = f'{5 * i}% shortcut')

plt.xticks(x)
plt.xlabel('Sensitivity to Shortcuts')
plt.ylabel('Shortcut Level')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14, 10)
fig.savefig('sensitivity.png', dpi = 200)

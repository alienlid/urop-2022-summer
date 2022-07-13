import matplotlib.pyplot as plt
import numpy as np

iid_fn = np.load('gaussian_blur/iid_fn.npy')
ood_fn = np.load('gaussian_blur/ood_fn.npy')
iid_ll = np.load('gaussian_blur/iid_ll.npy')
ood_ll = np.load('gaussian_blur/ood_ll.npy')

x = np.array(range(1, 6))
color = plt.cm.rainbow(np.linspace(0, 1, 21))

for i in range(21):
	plt.plot(x, iid_ll[i], label = f'{5 * i}% shortcut', c = color[i])

plt.xticks(x)
plt.xlabel('Severity')
plt.ylabel('IID accuracy, last-layer')
plt.legend()
fig = plt.gcf()
fig.set_size_inches(14, 10)
fig.savefig('iid_ll.png', dpi = 200)

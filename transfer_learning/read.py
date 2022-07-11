import numpy as np

f = open('gaussian_blur/full-network', 'r')
lines = f.readlines()

iid_fn = np.empty([21, 5])
ood_fn = np.empty([21, 5])
iid_ll = np.empty([21, 5])
ood_ll = np.empty([21, 5])

for k in range(105):
	comma = lines[7 * k + 4].find(',')
	i = int(lines[7 * k + 4][10 : comma]) / 5
	new_line = lines[7 * k + 4].find('\n')
	j = int(lines[7 * k + 4][comma + 12 : new_line])
	percent = lines[7 * k + 5].find('%')
	iid_fn[i][j] = float(lines[7 * k + 5][14 : percent]) / 100
	percent = lines[7 * k + 6].find('%')
	ood_fn[i][j] = float(lines[7 * k + 6][14 : percent]) / 100

np.save('gaussian_blur/iid_fn.npy', iid_fn.astype(np.uint8))
np.save('gaussian_blur/ood_fn.npy', ood_fn.astype(np.uint8))

f.close()

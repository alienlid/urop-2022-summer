import numpy as np

# ~ iid_fn = np.zeros([21, 5])
# ~ ood_fn = np.zeros([21, 5])
# ~ iid_ll = np.zeros([21, 5])
# ~ ood_ll = np.zeros([21, 5])

# ~ f = open('gaussian_blur/full-network.txt', 'r')
# ~ lines = f.readlines()

# ~ for k in range(105):
	# ~ comma = lines[7 * k + 4].find(',')
	# ~ i = int(lines[7 * k + 4][10 : comma]) // 5
	# ~ new_line = lines[7 * k + 4].find('\n')
	# ~ j = int(lines[7 * k + 4][comma + 12 : new_line]) - 1
	# ~ percent = lines[7 * k + 5].find('%')
	# ~ iid_fn[i, j] = float(lines[7 * k + 5][14 : percent]) / 100
	# ~ percent = lines[7 * k + 6].find('%')
	# ~ ood_fn[i, j] = float(lines[7 * k + 6][14 : percent]) / 100

# ~ np.save('gaussian_blur/iid_fn.npy', iid_fn.astype(np.float32))
# ~ np.save('gaussian_blur/ood_fn.npy', ood_fn.astype(np.float32))

# ~ f.close()

# ~ f = open('gaussian_blur/last-layer.txt', 'r')
# ~ lines = f.readlines()

# ~ for k in range(105):
	# ~ comma = lines[7 * k + 4].find(',')
	# ~ i = int(lines[7 * k + 4][10 : comma]) // 5
	# ~ new_line = lines[7 * k + 4].find('\n')
	# ~ j = int(lines[7 * k + 4][comma + 12 : new_line]) - 1
	# ~ percent = lines[7 * k + 5].find('%')
	# ~ iid_ll[i, j] = float(lines[7 * k + 5][14 : percent]) / 100
	# ~ percent = lines[7 * k + 6].find('%')
	# ~ ood_ll[i, j] = float(lines[7 * k + 6][14 : percent]) / 100

# ~ np.save('gaussian_blur/iid_ll.npy', iid_ll.astype(np.float32))
# ~ np.save('gaussian_blur/ood_ll.npy', ood_ll.astype(np.float32))

# ~ f.close()

fn = np.zeros([5, 21])
ll = np.zeros([5, 21])

f = open('gaussian_blur/s.txt', 'r')
lines = f.readlines()

for k in range(105):
	comma = lines[7 * k + 3].find(',')
	i = int(lines[7 * k + 3][10 : comma]) // 5
	new_line = lines[7 * k + 3].find('\n')
	j = int(lines[7 * k + 3][comma + 12 : new_line]) - 1
	fn[j, i] = float(lines[7 * k + 5])
	ll[j, i] = float(lines[7 * k + 6])
	
np.save('gaussian_blur/sensitivity_fn.npy', fn.astype(np.float32))
np.save('gaussian_blur/sensitivity_ll.npy', ll.astype(np.float32))

f.close()
	

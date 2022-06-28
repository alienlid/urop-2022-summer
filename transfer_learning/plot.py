import matplotlib.pyplot as plt
import numpy as np

# ~ IID = [
	# ~ [0.9598, 0.9550, 0.9533, 0.9488, 0.9389],
	# ~ [0.9695, 0.9637, 0.9624, 0.9561, 0.9475],
	# ~ [0.9745, 0.9726, 0.9676, 0.9653, 0.9624],
	# ~ [0.9846, 0.9828, 0.9822, 0.9767, 0.9734],
	# ~ ]
	
IID = [
	[0.9589, 0.9580, 0.9553, 0.9511, 0.9472, 0.9390],
	[0.9623, 0.9618, 0.9590, 0.9557, 0.9521, 0.9399],
	[0.9636, 0.9644, 0.9581, 0.9569, 0.9537, 0.9423],
	[0.9664, 0.9645, 0.9625, 0.9572, 0.9544, 0.9458],
	[0.9689, 0.9640, 0.9621, 0.9616, 0.9613, 0.9490],
	[0.9684, 0.9695, 0.9640, 0.9644, 0.9592, 0.9530],
	[0.9724, 0.9713, 0.9672, 0.9621, 0.9612, 0.9539],
	[0.9740, 0.9749, 0.9688, 0.9650, 0.9638, 0.9563],
	[0.9739, 0.9778, 0.9720, 0.9703, 0.9669, 0.9617],
	[0.9783, 0.9745, 0.9757, 0.9755, 0.9699, 0.9602],
	[0.9819, 0.9805, 0.9784, 0.9763, 0.9717, 0.9672],
	[0.9831, 0.9827, 0.9810, 0.9767, 0.9759, 0.9707],
	[0.9854, 0.9843, 0.9829, 0.9821, 0.9820, 0.9758]
	]
	
OOD = [
	[0.9600, 0.9598, 0.9304, 0.8996, 0.8719, 0.8238],
	[0.9634, 0.9622, 0.9405, 0.9045, 0.8812, 0.7953],
	[0.9642, 0.9637, 0.9448, 0.9112, 0.8892, 0.8276],
	[0.9669, 0.9650, 0.9479, 0.9180, 0.8964, 0.8536],
	[0.9684, 0.9672, 0.9506, 0.9322, 0.8635, 0.8535],
	[0.9697, 0.9687, 0.9523, 0.9204, 0.9161, 0.8614],
	[0.9706, 0.9715, 0.9573, 0.9353, 0.9248, 0.8556],
	[0.9738, 0.9730, 0.9589, 0.9452, 0.9276, 0.9055],
	[0.9757, 0.9738, 0.9664, 0.9498, 0.9321, 0.9071],
	[0.9776, 0.9769, 0.9697, 0.9590, 0.9414, 0.9245],
	[0.9792, 0.9807, 0.9703, 0.9589, 0.9550, 0.9351],
	[0.9812, 0.9836, 0.9754, 0.9638, 0.9653, 0.9381],
	[0.9854, 0.9834, 0.9792, 0.9731, 0.9631, 0.9570]
	]

x = np.array(range(6))

# ~ plt.plot(x, IID[0], label = '20% shortcut')
# ~ plt.plot(x, IID[1], label = '25% shortcut')
# ~ plt.plot(x, IID[2], label = '30% shortcut')
# ~ plt.plot(x, IID[3], label = '35% shortcut')
# ~ plt.plot(x, IID[4], label = '40% shortcut')
# ~ plt.plot(x, IID[5], label = '45% shortcut')
# ~ plt.plot(x, IID[6], label = '50% shortcut')
# ~ plt.plot(x, IID[7], label = '55% shortcut')
# ~ plt.plot(x, IID[8], label = '60% shortcut')
# ~ plt.plot(x, IID[9], label = '65% shortcut')
# ~ plt.plot(x, IID[10], label = '70% shortcut')
# ~ plt.plot(x, IID[11], label = '75% shortcut')
# ~ plt.plot(x, IID[12], label = '80% shortcut')
plt.plot(x, OOD[0], label = '20% shortcut')
plt.plot(x, OOD[1], label = '25% shortcut')
plt.plot(x, OOD[2], label = '30% shortcut')
plt.plot(x, OOD[3], label = '35% shortcut')
plt.plot(x, OOD[4], label = '40% shortcut')
plt.plot(x, OOD[5], label = '45% shortcut')
plt.plot(x, OOD[6], label = '50% shortcut')
plt.plot(x, OOD[7], label = '55% shortcut')
plt.plot(x, OOD[8], label = '60% shortcut')
plt.plot(x, OOD[9], label = '65% shortcut')
plt.plot(x, OOD[10], label = '70% shortcut')
plt.plot(x, OOD[11], label = '75% shortcut')
plt.plot(x, OOD[12], label = '80% shortcut')
plt.xticks(x)
plt.xlabel('Severity')
plt.ylabel('OOD accuracy')
plt.legend()
plt.show()

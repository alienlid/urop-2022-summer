import matplotlib.pyplot as plt
import numpy as np

IID = [
	[0.9598, 0.9550, 0.9533, 0.9488, 0.9389],
	[0.9695, 0.9637, 0.9624, 0.9561, 0.9475],
	[0.9745, 0.9726, 0.9676, 0.9653, 0.9624],
	[0.9846, 0.9828, 0.9822, 0.9767, 0.9734],
	]

x = np.array(range(1, 6))

plt.plot(x, IID[0], label = '20% shortcut')
plt.plot(x, IID[1], label = '40% shortcut')
plt.plot(x, IID[2], label = '60% shortcut')
plt.plot(x, IID[3], label = '80% shortcut')
plt.xticks(x)
plt.xlabel('Severity')
plt.ylabel('IID accuracy')
plt.legend()
plt.show()

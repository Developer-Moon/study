import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x) :
    return 1 / (1 + np.exp(-x)) # 0과 1사이

sigmoid2 = lambda x : 1 / (1 + np.exp(-x))

x = np.arange(-5, 5, 0.1)

print(x)
print(len(x)) # 100

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()
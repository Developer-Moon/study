import matplotlib.pyplot as plt
import numpy as np


def tanh(x) :
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)            # tanh : -1 ~ 1 사이로

plt.plot(x, y)
plt.grid()
plt.show()


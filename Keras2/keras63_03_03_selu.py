import matplotlib.pyplot as plt
import numpy as np

def selu(x) :
    return np.maximum(-1, x)

selu2 = lambda x : np.maximum(-1, x)

x = np.arange(-5, 5, 0.1)
y = selu2(x)

plt.plot(x, y)
plt.grid()
plt.show()



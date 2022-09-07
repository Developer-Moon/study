import matplotlib.pyplot as plt
import numpy as np

def elu(x) :
    return np.maximum(np.exp(-x)-1, x)

elu2 = lambda x : np.maximum(-1, x)

x = np.arange(-5, 5, 0.1)
y = elu2(x)

plt.plot(x, y)
plt.grid()
plt.show()
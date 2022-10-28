import matplotlib.pyplot as plt
import numpy as np

def Leaky(x):
    return np.maximum(0.01 * x, x)

reaky = lambda x : np.maximum(0.01 * x, x)

x = np.arange(-5, 5, 0.1)
y = reaky(x)

plt.plot(x, y)
plt.grid()
plt.show()


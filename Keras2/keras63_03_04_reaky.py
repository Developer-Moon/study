import matplotlib.pyplot as plt
import numpy as np

def reaky(x) :
    return np.maximum(-1, x)

reaky = lambda x : np.maximum(-1, x)

x = np.arange(-5, 5, 0.1)
y = reaky(x)

plt.plot(x, y)
plt.grid()
plt.show()


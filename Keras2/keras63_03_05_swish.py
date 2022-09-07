import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


x = np.arange(-5, 5, 0.1)
y = tf.keras.activations.swish(x)

plt.plot(x, y)
plt.grid()
plt.show()


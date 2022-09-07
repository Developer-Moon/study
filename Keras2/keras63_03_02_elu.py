import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf



x = np.arange(-5, 5, 0.1)
y = tf.keras.activations.elu(x)

plt.plot(x, y)
plt.grid()
plt.show()

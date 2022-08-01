import numpy as np

a = np.array([1, 2, 3])

t = ['t1', 't2']

t = np.array(t)

np.tile(a, (2, 2))

print(np.tile(a, (2, t)).shape)



import numpy as np   
"""
a = np.array(range(1, 11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)        #(6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) #(6, 4)(6.)
"""

a = np.array(range(1, 11))
size = 5

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape)        #(6, 5)

x = bbb[:, :-1]
y = bbb[:, -1]
print(x, y)
print(x.shape, y.shape) #(6, 4)(6.)


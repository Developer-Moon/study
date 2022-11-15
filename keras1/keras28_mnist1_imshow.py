import numpy as np   
from tensorflow.keras.datasets import mnist    # 이거 활성화되게 돌려놓기

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)  y라벨이 없다면 

print(x_train[0]) 
print(y_train[0]) # 5

import matplotlib.pyplot as plt        
plt.imshow(x_train[5], 'gray')
plt.show()
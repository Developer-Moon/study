# 증폭해서 npy에 저장
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(x_test.shape)  # (60000, 28, 28)
print(y_test.shape)  # (60000,)

augument_size=5000
randinx = np.random.randint(x_train.shape[0])


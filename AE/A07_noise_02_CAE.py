from enum import auto
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, UpSampling2D
from keras.datasets import mnist
import numpy as np


(x_train, _), (x_test, _) = mnist.load_data() 
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.  # /255. 스케일링 작업
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # 정규분표형태로
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)    

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) # 0이하는 0, 1이상은 1로 
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1) 
 

def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(1, 1), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(29, 29), activation='relu'))
    return model

model = autoencoder(hidden_layer_size=64)
# model = autoencoder(hidden_layer_size=154) # PCA의 95% 성능
# model = autoencoder(hidden_layer_size=331)   # PCA의 95% 성능

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.fit(x_train_noised, x_train, epochs=2) # 노이즈가 없는걸 x 노이즈가 없는걸 y

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])
    
# 노이즈가 들어간 이미지     
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel('NOISE', size=20)
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])    


# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]) :
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()    
plt.show()

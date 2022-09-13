# [실습] 4번 카피 복붙
# CNN으로 딥하게 구성
# Upsampling을 찾아서 이해하고 반드시 추가할 것!! GAN에서 사용하게 될 것 
# MaxPooling은 Downsampling이다

# Upsampling이란

# 니얼리스트 
# 쌍선형 보관법, 보간법?

from enum import auto
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, UpSampling2D
from keras.datasets import mnist
import numpy as np


(x_train, _), (x_test, _) = mnist.load_data() 
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.  # /255. 스케일링 작업
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.



def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(1, 1), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=(29, 29), activation='sigmoid'))
    model.summary()
    return model



model = autoencoder(hidden_layer_size=1) # PCA의 95% 성능

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20, 7))

# 이미지 5개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

#원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel('INPUT', size=20)
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel('OUTPUT', size=20)
    ax.grid(False)    
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()    
plt.show()

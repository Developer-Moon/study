from enum import auto
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.datasets import mnist
import numpy as np
# autoencoder는 input과 output이 똑같다
# 특성인 부분은 다시 나오고, 아닌 속성은 버린다[원본을 재조합한다]


# (x_train, y_train), (x_test, y_test) = mnist.load_data() 
(x_train, _), (x_test, _) = mnist.load_data() 
x_train = x_train.reshape(60000, 784).astype('float32')/255. # .은 부동소수점 형태로 하겠다
x_test = x_test.reshape(10000, 784).astype('float32')/255.

input_img = Input(shape=(784,))                      
encoded = Dense(64, activation='relu')(input_img)     # 필요없는 부분을 줄이는 과정
# encoded = Dense(1064, activation='relu')(input_img) # 노드를 늘릴경우? - 조금 더 좋아졌다
# encoded = Dense(16, activation='relu')(input_img)   # 노드를 줄일경우? - 형편없이 안 좋아진다
# decoded = Dense(784, activation='relu')(encoded)    # loss='mse' - 안좋다
# decoded = Dense(784, activation='linear')(encoded)  # loss='mse' - 안좋다
# decoded = Dense(784, activation='tanh')(encoded)    # loss='mse' - 안좋다
decoded = Dense(784, activation='sigmoid')(encoded)    

autoencoder = Model(input_img, decoded)


autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) # 이미지 생성관련해서는 acc가 많이 중요하지 않다

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2) # 준지도학습[지도도 아니고 비지도도 아니다] x는 x다 라는 개념

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20, 4))
for i in range(n) :
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

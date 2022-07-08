from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D      
from tensorflow.keras.datasets import mnist
import numpy as np                            

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,) 
# reshape할때 모든 객체의 곱은 같아야 한다, 순서는 바뀌면 안되지만 모양만 바꾸면 된다

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(np.unique(y_train, return_counts=True)) #  y값의 라벨이 먼지 확인해야한다
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

# one hot하기

# 그려봐 맹
# acc 0.98이상
# Conv2D layer 3개이상 쌓아라

"""
# CNN모델                                                                                         
model = Sequential()                                                                             
model.add(Conv2D(filters=64, kernel_size=(3,3),
                 padding='same',           # padding 0이라는 패딩을 씌워서 이미지를 조각낼때 가장자리 부분을 두번 이상 넣어줘서 다른 부분보다 덜 학습되는걸 방지 
                 input_shape=(28, 28, 1)))  # 통상 shape를 다음 레이어에도 유지하고 싶을때 padding을 쓴다                                                                                                         
model.add(MaxPooling2D())    #(14, 14, 64) 
model.add(Conv2D(32, (2, 2),
                 padding='valid',          # 디폴트
                 activation='relu'))                          
model.add(Flatten()) # (N, 252)                                                                         
model.add(Dense(32, activation='relu'))                                                                                                         
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary() 
"""




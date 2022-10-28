from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import Conv1D, LSTM, Reshape
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np                            
import pandas as pd 
from keras.datasets import mnist

import tensorflow as tf            
tf.random.set_seed(66)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)           
print(x_test.shape, y_test.shape)            
                                              
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape)                          
print(np.unique(y_train, return_counts=True)) 

from tensorflow.python.keras.utils.np_utils import to_categorical # 범주
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델구성                                                                                    
model = Sequential()                                                                             
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())                                          # (N, 14, 14, 64)                                                                    
model.add(Conv2D(32, (3,3), padding='valid', activation='relu'))   # (N, 12, 12, 32)  
model.add(Conv2D(7, (3,3)))                                        # (N, 10, 10, 7)   
model.add(Flatten())                                               # (N, 700) 1차원으로 펴서 10 x 10 x 77                       
model.add(Dense(100, activation='relu'))                           # (N, 100)


model.add(Reshape(target_shape=(100,1,1)))                         # Reshape 모양만 바꿔준다(순서와 내용은 바뀌지 않는다) 연산량=0


model.add(Conv2D(10, 1))                                           # (N, 98, 10) 
model.add(Flatten())                                               # (N, 16)
model.add(Dense(32, activation='relu'))                            # (N, 32)                                                 
model.add(Dense(10, activation='softmax'))                         # (N, 10)
model.summary() 

"""
ayer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 64)        640
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 12, 12, 32)        18464
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 10, 7)         2023
_________________________________________________________________
flatten (Flatten)            (None, 700)               0
_________________________________________________________________
dense (Dense)                (None, 100)               70100
_________________________________________________________________
reshape (Reshape)            (None, 100, 1, 1)         0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 100, 1, 10)        20
_________________________________________________________________
flatten_1 (Flatten)          (None, 1000)              0
_________________________________________________________________
dense_1 (Dense)              (None, 32)                32032
_________________________________________________________________
dense_2 (Dense)              (None, 10)                330
=================================================================
Total params: 123,609
Trainable params: 123,609
Non-trainable params: 0    
"""

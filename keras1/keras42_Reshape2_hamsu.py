from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.python.keras.layers import Conv1D, LSTM, Reshape
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np                            
from keras.datasets import mnist

import tensorflow as tf            
tf.random.set_seed(66)



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)           # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)             # (10000, 28, 28) (10000,) 
                                              # reshape할때 모든 객체의 곱은 같아야 한다, 순서는 바뀌면 안되지만 모양만 바꾸면 된다



x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(np.unique(y_train, return_counts=True)) #  y값의 라벨이 먼지 확인해야한다
                                              # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
                                              # array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))


from tensorflow.python.keras.utils.np_utils import to_categorical # 범주
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(pd.get_dummies(y_train))
print(y_train.shape)                          # (60000, 10)



#2. 모델구성CNN모델    
input_01 = Input(shape=(28, 28, 1))
conv2D_01 = Conv2D(64, kernel_size=(3,3), padding='same')(input_01)
maxplooling2D_01 = MaxPooling2D()(conv2D_01)
conv2D_02 = Conv2D(32, (3,3), padding='valid', activation='relu')(maxplooling2D_01)
conv2D_03 = Conv2D(7, (3,3))(conv2D_02)
flatten_01 = Flatten()(conv2D_03)
dense_01 = Dense(32)(flatten_01)
dense_02 = Dense(100)(dense_01)
reshape_01 = Reshape(target_shape=(100,1,1))(dense_02)
conv2D_04 = Conv2D(10, 1)(reshape_01)
flatten_02 = Flatten()(conv2D_04)
dense_03 = Dense(32, activation='relu')(flatten_02)
dense_04 = Dense(10)(dense_03)
output_01 = Dense(10, activation='softmax')(dense_04)
model = Model(inputs=input_01, outputs=output_01)
model.summary()


"""                                                                                                     
model = Sequential()                                                                             
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28, 28, 1))) 
model.add(MaxPooling2D())                                          # (N, 14, 14, 64)                                                                    
model.add(Conv2D(32, (3,3), padding='valid', activation='relu'))   # (N, 12, 12, 32)  
model.add(Conv2D(7, (3,3)))                                        # (N, 10, 10, 7)                    
model.add(Reshape(target_shape=(16,1)))                            # Reshape 모양만 바꿔준다(순서와 내용은 바뀌지 않는다) 연산량=0                                              # (N, 16)
model.add(Dense(32, activation='relu'))                            # (N, 32)                                                 
model.add(Dense(10, activation='softmax'))                         # (N, 10)
model.summary() 
"""

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, Conv1D      
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np                            
import pandas as pd 
from keras.datasets import mnist
from keras.layers import GlobalAveragePooling2D

import tensorflow as tf            
tf.random.set_seed(66)

# acc 0.98이상

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)           # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)             # (10000, 28, 28) (10000,) 
                                              # reshape할때 모든 객체의 곱은 같아야 한다, 순서는 바뀌면 안되지만 모양만 바꾸면 된다





x_train = x_train.reshape(60000, 784, 1)
x_test = x_test.reshape(10000, 784, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(np.unique(y_train, return_counts=True)) #  y값의 라벨이 먼지 확인해야한다
                                              # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
                                              # array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))


# from tensorflow.python.keras.utils.np_utils import to_categorical # 범주
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape,x_test.shape)
x_train = x_train.reshape(60000,28*28,1)
x_test = x_test.reshape(10000,28*28,1)

#2.모델구성
model = Sequential()
model.add(Conv1D(8,2,activation='relu', input_shape=(28*28,1))) 
model.add(Flatten())
model.add(Dense(6,activation= 'relu'))
model.add(Dense(4,activation= 'relu'))
model.add(Dense(2,activation= 'relu'))
model.add(Dense(10,activation='softmax'))


#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
es= EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auto', verbose=1, factor=0.5)
tb = TensorBoard(log_dir='D:/study_data/tensorboard_log/_graph', histogram_freq=0, write_graph=True, write_images=True)
# 실행방법 : tensorboard --logdir=. (경로)
# http://localhost:6006
# http://127.0.0.1:6006 아이피

learning_rate=0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer ,metrics=['acc'])

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[es, reduce_lr, tb], validation_split=0.2)
end = time.time() - start

loss, acc = model.evaluate(x_test, y_test)
print('learning_rate :', learning_rate)
print('Loss :', round(loss, 4))
print('Acc :', round(acc, 4))
print('Time :', round(end, 4))


##시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))


#1. 
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2
plt.subplot(2, 1, 2)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()

# learning_rate : 0.01
# Loss : 2.3013
# Acc : 0.1135
# Time : 157.097
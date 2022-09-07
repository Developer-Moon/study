from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input      
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


x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape) # (60000, 28, 28, 1)
print(np.unique(y_train, return_counts=True)) #  y값의 라벨이 먼지 확인해야한다
                                              # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
                                              # array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))


# from tensorflow.python.keras.utils.np_utils import to_categorical # 범주
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)




# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(pd.get_dummies(y_train))
print(x_train.shape, x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)                         
print(y_train.shape, y_test.shape) # (60000, 10) (10000, 10)               





#2. 모델구성CNN모델                                                                                         
model = Sequential()                                                                             
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(28, 28, 1)))
                                              # padding 0이라는 패딩을 씌워서 이미지를 조각낼때 가장자리 부분을 두번 이상 넣어줘서 다른 부분보다 덜 학습되는걸 방지 
                                              # 통상 shape를 다음 레이어에도 유지하고 싶을때 padding을 쓴다                                                                                    
model.add(Conv2D(32, (2, 2), padding='valid', activation='relu'))    # padding='valid' 디폴트
model.add(Conv2D(32, (2, 2), activation='relu'))    
model.add(Conv2D(32, (2, 2), activation='relu'))   
model.add(MaxPooling2D())                     # MaxPooling2D을 했을때 윗단의 크기에서 반으로 줄이고 큰 값만 살린다
model.add(Flatten())                          # (N, 252)   Flatten을 안써도 하단 dense로 계산된다                                                             
model.add(Dense(32, activation='relu'))    
model.add(Dropout(0.2))                                                                                                     
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
es= EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=7, mode='auto', verbose=1, factor=0.5)

learning_rate=0.01
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer ,metrics=['acc'])

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[es, reduce_lr], validation_split=0.2)
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


# 훈련시 
# 938/1500 [=================>............] - ETA: 2s - loss: 0.8130 - acc: 0.7967  
# 1500은 x data 60000개중 validation 0.2 를 뺀 48000에서 batch_size 32를 나누면 1epoch당 1500이 나온다
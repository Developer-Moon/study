from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,Dropout   
from tensorflow.python.keras.models import Sequential
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
import pandas as pd
import numpy as np

import tensorflow as tf            
tf.random.set_seed(66)

#성능은 CNN 보다 좋게

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

print(np.unique(y_train,return_counts=True))

# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))
print(y_train.shape) #(50000, 100)
print(y_test.shape) #(10000, 10)

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 

# y_train = pd.get_dummies((y_train)) 
# y_test = pd.get_dummies((y_test))
#get_dummies 사용 시 오류가 뜨는 이유는 get_dummies는 2차원 타입의 데이터까지만  인코딩 가능하기 때문이다.
#3차원 이상의 데이터를 인코딩할 때는 원핫인코딩과 카테고리컬을 사용한다.
print(y_train.shape) #(50000, 100)
print(y_test.shape) #(10000, 100)

#2. 모델구성
model = Sequential()

# model.add(Dense(64, input_shape=(28*28, ))) - 28x28 사이즈였다는걸 이런식으로 명시도 가능 
model.add(Dense(64, input_shape=(32*32*3, )))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(100, activation='softmax'))




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint      
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)        

  
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  
                                            # batch_size:32 디폴트값 3번정도 말 한듯




#4. 결과, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 


from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)

y_test = tf.argmax(y_test, axis=1)
print(y_test)


acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)



# loss :  4.605496406555176
# accuracy :  0.010300000198185444
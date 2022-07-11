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
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

print(np.unique(y_train,return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],dtype=int64))

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)


# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(pd.get_dummies(y_train))
# print(y_train.shape)    

from tensorflow.keras.utils import to_categorical 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 




print(y_train.shape) #(50000, 10)
print(y_test.shape) #(10000, 10)





#2. 모델구성
model = Sequential()

# model.add(Dense(64, input_shape=(28*28, ))) - 28x28 사이즈였다는걸 이런식으로 명시도 가능 
model.add(Dense(64, input_shape=(28*28, )))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))





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

# CNN 사용
# loss :  0.05932530015707016
# accuracy :  0.985099971294403


# DNN 사용
# loss :  0.05968634784221649
# accuracy :  0.9843999743461609
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, LSTM  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import numpy as np                            
import pandas as pd 
from keras.datasets import mnist

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


from tensorflow.python.keras.utils.np_utils import to_categorical # 범주
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)
# print(pd.get_dummies(y_train))
print(y_train.shape)                          # (60000, 10)





#2. 모델구성CNN모델     
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(784, 1))) # 최대넓이가 가로13 세로 1이라 커널 사이즈 최대가 (1, 1)이 된다                                                                          
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))






#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint      
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)        

  
hist = model.fit(x_train, y_train, epochs=10, batch_size=500, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  
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

# loss :  0.05932530015707016
# accuracy :  0.9850999712944031

# LSTM
# loss :  nan
# accuracy :  0.09799999743700027
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D ,Dropout   
from tensorflow.python.keras.models import Sequential
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
import pandas as pd
import numpy as np

import tensorflow as tf            
tf.random.set_seed(66)

#성능은 CNN 보다 좋게 - 3차원 데이터를 2차원으로 만들어야 하는 경우가 생길때 이렇게 쓴다고 한다 

# img모델을 만들때는 CNN, DNN 2가지로 만들 수 있다

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)           # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)             # (10000, 28, 28) (10000,) 
                                              # reshape할때 모든 객체의 곱은 같아야 한다, 순서는 바뀌면 안되지만 모양만 바꾸면 된다

print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
                                              # array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))


x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
print(x_train.shape) # (60000, 784)
print(y_train.shape) # (60000,)
 #  y값의 라벨이 먼지 확인해야한다
                                              

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(pd.get_dummies(y_train))
print(y_train.shape) # (60000, 10)  



#2. 모델구성
model = Sequential()

# model.add(Dense(64, input_shape=(28*28, ))) - 28x28 사이즈였다는걸 이런식으로 명시도 가능 
model.add(Dense(64, input_shape=(784, )))
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
# loss :  0.15213651955127716
# accuracy :  0.9664000272750854
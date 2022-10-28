import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,Dropout,Input,LSTM,Conv1D
from tensorflow.python.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(60000,28*28*1)
x_test = x_test.reshape(10000,28*28*1)
print(x_train.shape)

import numpy as np
print(np.unique(y_train,return_counts=True))
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,x_test.shape)

x_train = x_train.reshape(60000,784,1)
x_test = x_test.reshape(10000,784,1)

#2.모델구성
model = Sequential()
model.add(Conv1D(100,2,activation='relu', input_shape=(784,1))) 
model.add(Flatten())
model.add(Dense(50,activation= 'relu'))
model.add(Dense(50,activation= 'relu'))
model.add(Dense(50,activation= 'relu'))
model.add(Dense(50,activation= 'relu'))
model.add(Dense(10,activation='softmax'))

#3.컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam')
ES = EarlyStopping(monitor='val_loss', patience=10, mode='min')
model.fit(x_train, y_train, epochs=1, batch_size=1000,validation_split=0.1)

print(y_test)

#4.평가 훈련
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print(':acc스코어 ', acc)

#CNN
# :acc스코어  0.7355

#함수
# :acc스코어  0.8635

#LSTM
# :acc스코어  0.1049

#Conv1D
#:acc스코어  0.6254
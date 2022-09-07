from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16, VGG19
from keras.datasets import cifar100
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = x_train.reshape(50000,32*32*3)
x_test = x_test.reshape(10000,32*32*3)
print(x_train.shape, x_test.shape )
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
 
x_train = x_train.reshape(50000, 32,32, 3)
x_test = x_test.reshape(10000,32,32, 3)


print(x_train.shape)
print(y_train.shape)
print(y_train[:5])

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.transform(y_test).toarray()

x_train = x_train.reshape(50000,32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)



#2. 모델구성
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# vgg16.trainable=False # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100, activation='softmax'))


#3. 컴파일
from tensorflow.keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(lr=learning_rate)

model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau # 

es= EarlyStopping(monitor='val_loss', patience=20, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) # factor=0.5 : 50% 만큼 lr을 감소 시킨다  디폴트 lr은 0.001
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4, batch_size=128, callbacks=[es, reduce_lr])
end = time.time() - start


#4.결과, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc스코어: ', acc)

# vgg16.trainable=False - acc스코어:  0.1
# vgg16.trainable=True - 


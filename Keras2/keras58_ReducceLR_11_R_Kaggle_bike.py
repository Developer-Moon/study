from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import time


#1. Data
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True)
test_set.drop('datetime',axis=1,inplace=True)
train_set.drop('casual',axis=1,inplace=True) 
train_set.drop('registered',axis=1,inplace=True) 

x = train_set.drop(['count'], axis=1)
y = train_set['count']

print(x.shape, y.shape)                 # (10886, 12) (10886,)
print(np.unique(y, return_counts=True))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. Model
activation='relu'
drop=0.2
optimizer='adam'

inputs = Input(shape=(12), name='input')
x = Dense(64, activation=activation, name='hidden1')(inputs)
x = Dense(128, activation=activation, name='hidden2')(x)
x = Dense(128, activation=activation, name='hidden3')(x)
x = Dense(128, activation=activation, name='hidden4')(x)
outputs = Dense(1, activation=activation)(x)
model = Model(inputs=inputs, outputs=outputs)


#3. Compile
model.compile(optimizer=optimizer, metrics=['mse'], loss='mse')
es= EarlyStopping(monitor='val_loss', patience=50, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', verbose=1, factor=0.5) # factor=0.5 : 50% 만큼 lr을 감소 시킨다  디폴트 lr은 0.001
start = time.time()
model.fit(x_train, y_train, epochs=500, validation_split=0.4, batch_size=128, callbacks=[es, reduce_lr])
end = time.time() - start


#4. Result
y_pred = model.predict(x_test)
loss = model.evaluate(x_test, y_test)
r2 = r2_score(y_test, y_pred)

print('Time :', end)
print('Loss :', loss[0])
print('R2 :', r2)


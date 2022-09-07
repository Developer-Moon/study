from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
import time


#1. Data
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) 
y = np.array(y).reshape(-1, 1)

print(x.shape, y.shape)                 # (891, 7) (891, 1)    
print(np.unique(y, return_counts=True)) # (array([0, 1], dtype=int64), array([549, 342], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)
    
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


#2. Model
activation='relu'
drop=0.2
optimizer='adam'

inputs = Input(shape=(7), name='input')
x = Dense(64, activation=activation, name='hidden1')(inputs)
x = Dense(128, activation=activation, name='hidden2')(x)
x = Dense(128, activation=activation, name='hidden3')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)


#3. Compile
model.compile(optimizer=optimizer, metrics=['acc'], loss='binary_crossentropy')
es= EarlyStopping(monitor='val_loss', patience=50, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto', verbose=1, factor=0.5) # factor=0.5 : 50% 만큼 lr을 감소 시킨다  디폴트 lr은 0.001
start = time.time()
model.fit(x_train, y_train, epochs=500, validation_split=0.4, batch_size=128, callbacks=[es, reduce_lr])
end = time.time() - start


#4. Result
loss, acc = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print('Time :', end)
print('Loss :', loss)
print('Acc :', acc)

# Time : 14.490068674087524
# Loss : 0.5402230024337769
# Acc : 0.7765362858772278

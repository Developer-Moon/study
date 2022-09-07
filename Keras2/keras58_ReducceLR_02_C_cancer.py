from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import numpy as np
import time


#1. Data
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)                 # (569, 30) (569,)
print(np.unique(y, return_counts=True)) # (array([0, 1]), array([212, 357], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=123)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. Model
activation='relu'
drop=0.2
optimizer='adam'

inputs = Input(shape=(30), name='input')
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

# Time : 18.185415506362915
# Loss : 0.16394253075122833
# Acc : 0.9790209531784058

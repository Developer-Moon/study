from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D 
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score 
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target 
print(x.shape, y.shape) # (506, 13) (506,)




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

print(x_train.shape) # (404, 13)
print(x_test.shape)  # (102, 13)

x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)


#2. 모델구성
model = Sequential()
model.add(Conv1D(200,2,activation='relu', input_shape=(13,1))) 
model.add(Flatten())
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')             
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint  
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=300, batch_size=100, verbose=1, callbacks=[earlyStopping], validation_split=0.2)




# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# loss :  11.482247352600098
# r2스코어 :  0.8626245121807405

# loss :  13.23200798034668
# r2스코어 :  0.8416900670657803


# Conv1D
# loss :  10.78105354309082
# r2스코어 :  0.8710136998567093
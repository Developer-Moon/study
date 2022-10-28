from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D 
from sklearn.model_selection import train_test_split  
from sklearn.datasets import fetch_california_housing   
from sklearn.metrics import r2_score 
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  


#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data  
y = datasets.target 

print(x.shape, y.shape) # (20640, 8) (20640,)



x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )

print(x_train.shape) # (16512, 8)
print(x_test.shape)  # (4128, 8)

x_train = x_train.reshape(16512, 8, 1)
x_test = x_test.reshape(4128, 8, 1)

#2. 모델구성
model = Sequential()
model.add(Conv1D(200,2,activation='relu', input_shape=(8,1))) 
model.add(Flatten())
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                              
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# loss :  [0.2673729360103607, 0.3355602025985718]   기존
# r2스코어 : 0.8082063054489048

# loss :  [0.9301130175590515, 0.6812400817871094] dropout
# r2스코어 : 0.33280505872906796


# Conv1D
# loss : 0.46005740761756897
# r2스코어 : 0.6699886349565283
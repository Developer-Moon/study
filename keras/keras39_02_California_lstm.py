from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input, LSTM
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

x = x.reshape(20640, 8, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )


#2. 모델구성
model = Sequential()                                                                             
model.add(LSTM(units=200, activation='relu', input_shape=(8, 1))) # 최대넓이가 가로13 세로 1이라 커널 사이즈 최대가 (1, 1)이 된다                                                                          
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
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

# CNN 모델
# loss :  [0.4866293966770172, 0.5076679587364197]
# r2스코어 : 0.6509277323826514

# LSTM사용
# loss :  [1.3940738439559937, 0.9365211725234985]
# r2스코어 : -6.229756058351299e-06
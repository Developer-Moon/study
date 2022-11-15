from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense  
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score 
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66)

scaler = MinMaxScaler()     
# scaler = StandardScaler() 
# scaler = MaxAbsScaler()                                                                                  
# scaler = RobustScaler()
scaler.fit(x_train)                    
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)  


#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()

model.save_weights('./_save/keras23_5_save_weights1_model.h5') # 랜덤 가중치


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')             
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)                                                   
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, callbacks=[earlyStopping], validation_split=0.2)



model.save_weights('./_save/keras23_5_save_weights2_model.h5') # 저장된 가중치

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) # loss :  49.679412841796875    val_loss: 65.4229


y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)












# loss    : 11.616869926452637
# r2스코어 : 0.8610138651001857
# 걸린시간 : 5.067552804946899

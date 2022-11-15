from sklearn.datasets import fetch_california_housing  
import numpy as np
from sklearn import metrics         
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler    
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input      
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score 
import time


#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data  
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )


scaler = StandardScaler()
scaler.fit(x_train)    
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)       



#2. 모델구성
input_01 = Input(shape=(8,))
dense_01 = Dense(50)(input_01)
dense_02 = Dense(70, activation='relu')(dense_01)
dense_03 = Dense(80, activation='relu')(dense_02)
dense_04 = Dense(50, activation='relu')(dense_03)
dense_05 = Dense(10, activation='relu')(dense_04)
dense_06 = Dense(10, activation='relu')(dense_05)
dense_07 = Dense(10, activation='relu')(dense_06)
dense_08 = Dense(10, activation='relu')(dense_07)
dense_09 = Dense(10, activation='relu')(dense_08)
dense_10 = Dense(10, activation='relu')(dense_09)
dense_11 = Dense(10, activation='relu')(dense_10)
dense_11 = Dense(10, activation='relu')(dense_10)
dense_12 = Dense(10, activation='relu')(dense_11)
dense_13 = Dense(10, activation='relu')(dense_12)
dense_14 = Dense(10, activation='relu')(dense_13)
dense_15 = Dense(10, activation='relu')(dense_14)
dense_16 = Dense(10, activation='relu')(dense_15)
dense_17 = Dense(10, activation='relu')(dense_16)
output_01 = Dense(1)(dense_17)
model = Model(inputs=input_01, outputs=output_01)

# model.save('./_save/keras23_7_02_California_01_save_model.h5')          
# model.save_weights('./_save/keras23_7_02_California_03_save_weights.h5')   
         
# model = load_model('./_save/keras23_7_02_California_01_save_model.h5')        
# model.load_weights('./_save/keras23_7_02_California_03_save_weights.h5')        
model.load_weights('./_save/keras23_7_02_California_04_save_weights.h5')   

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                              
# from tensorflow.python.keras.callbacks import EarlyStopping      
# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
# hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlyStopping], verbose=1)

# model.save('./_save/keras23_7_02_California_02_save_model.h5') 
# model.save_weights('./_save/keras23_7_02_California_04_save_weights.h5')

# model = load_model('./_save/keras23_7_02_California_02_save_model.h5')  


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) 
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# loss :  0.3014218211174011, 0.3486739993095398] --------------------- 기존훈련        
# r2스코어 : 0.7837821188052695

# loss :  [0.283634752035141, 0.3506264388561249] --------------------- 01_save_model
# r2스코어 : 0.7965411687230853

# loss :  [0.3014218211174011, 0.3486739993095398] --------------------- 02_save_model : 모델과 가중치 같이 저장          [기존 훈련의 가장 좋은 값과 같다]
# r2스코어 : 0.7837821188052695

# loss :  [5.780257701873779, 2.0936949253082275] ---------------------- 03_save_weights : 랜덤 가중치 저장
# r2스코어 : -3.146332438414742

# loss :  [0.3014218211174011, 0.3486739993095398] --------------------- 04_save_weights : 훈련된 가장 좋게 저장된 가중치  [기존 훈련의 가장 좋은 값과 같다]
# r2스코어 : 0.7837821188052695

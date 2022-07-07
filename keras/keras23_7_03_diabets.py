import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_diabetes  
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input      
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 


#1. 데이터
datasets = load_diabetes()   
x = datasets.data
y = datasets.target      


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )


scaler = RobustScaler()
scaler.fit(x_train)     # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)       # train이 변한 범위에 맞춰서 변환됨




#2. 모델구성
input_01 = Input(shape=(10,)) 
dense_01 = Dense(10)(input_01)
dense_02 = Dense(100)(dense_01)
dense_03 = Dense(100)(dense_02)
dense_04 = Dense(100)(dense_03)
dense_05 = Dense(100)(dense_04)
dense_06 = Dense(100)(dense_05)
dense_07 = Dense(10)(dense_06)
output_01 = Dense(1)(dense_07)
model = Model(inputs=input_01, outputs=output_01)
model.summary()


# model.save('./_save/keras23_7_03_diabets_01_save_model.h5')          
# model.save_weights('./_save/keras23_7_03_diabets_03_save_weights.h5')   
         
# model = load_model('./_save/keras23_7_03_diabets_01_save_model.h5')        
# model.load_weights('./_save/keras23_7_03_diabets_03_save_weights.h5')        
model.load_weights('./_save/keras23_7_03_diabets_04_save_weights.h5')   

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                 
# from tensorflow.python.keras.callbacks import EarlyStopping      
# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
# hist = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)

# model.save('./_save/keras23_7_03_diabets_02_save_model.h5') 
# model.save_weights('./_save/keras23_7_03_diabets_04_save_weights.h5')

# model = load_model('./_save/keras23_7_03_diabets_02_save_model.h5')  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) 
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

# loss :  [3225.11865234375, 46.476810455322266] --------------------- 기존훈련        
# r2스코어 : 0.5030664612575949

# loss :  [3225.337158203125, 46.81227493286133] --------------------- 01_save_model
# r2스코어 : 0.5030328628921783

# loss :  [3225.11865234375, 46.476810455322266] --------------------- 02_save_model : 모델과 가중치 같이 저장          [기존 훈련의 가장 좋은 값과 같다]
# r2스코어 : 0.5030664612575949

# loss :  [29897.853515625, 152.921630859375] ---------------------- 03_save_weights : 랜덤 가중치 저장
# r2스코어 : -3.6067287664442453

# loss :  [3225.11865234375, 46.476810455322266] --------------------- 04_save_weights : 훈련된 가장 좋게 저장된 가중치  [기존 훈련의 가장 좋은 값과 같다]
# r2스코어 : 0.5030664612575949




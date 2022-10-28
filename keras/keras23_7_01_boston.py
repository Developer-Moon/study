from sklearn.datasets import load_boston   
from sklearn.model_selection import train_test_split   
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input                    
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 
import numpy as np       
from sklearn import metrics         
from sklearn.metrics import r2_score 

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )
                                                                       
scaler = MaxAbsScaler()
scaler.fit(x_train)                    
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)       



#2. 모델구성
input_01 = Input(shape=(13,))
dense_01 = Dense(14)(input_01)
dense_02 = Dense(20, activation='relu')(dense_01)
dense_03 = Dense(30, activation='relu')(dense_02)
dense_04 = Dense(20, activation='relu')(dense_03)
dense_05 = Dense(10, activation='relu')(dense_04)
output_01 = Dense(1)(dense_05)
model = Model(inputs=input_01, outputs=output_01)

# model.save('./_save/keras23_7_01_boston_01_save_model.h5')          
# model.save_weights('./_save/keras23_7_01_boston_03_save_weights.h5')   
         
# model = load_model('./_save/keras23_7_01_boston_01_save_model.h5')        
# model.load_weights('./_save/keras23_7_01_boston_03_save_weights.h5')        
model.load_weights('./_save/keras23_7_01_boston_04_save_weights.h5')   



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                                                                                                                                                                                                                
# from tensorflow.python.keras.callbacks import EarlyStopping      
# earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)          
# hist = model.fit(x_train, y_train, epochs=300, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  


# model.save('./_save/keras23_7_01_boston_02_save_model.h5') 
# model.save_weights('./_save/keras23_7_01_boston_04_save_weights.h5')

# model = load_model('./_save/keras23_7_01_boston_02_save_model.h5')  


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) 
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# loss :  10.419925689697266 --------------------- 기존훈련
# r2스코어 : 0.8753342904935599

# loss :  10.530672073364258 --------------------- 01_save_model
# r2스코어 : 0.87400929978287

# loss :  10.419925689697266 --------------------- 02_save_model : 모델과 가중치 같이 저장
# r2스코어 : 0.8753342904935599

# loss :  595.8762817382812 ---------------------- 03_save_weights : 랜덤 가중치 저장
# r2스코어 : -6.129161247973857

# loss :  10.419925689697266 --------------------- 04_save_weights : 훈련된 가장 좋게 저장된 가중치
# r2스코어 : 0.8753342904935599
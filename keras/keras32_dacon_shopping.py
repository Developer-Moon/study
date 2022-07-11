import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from scipy.stats import norm, skew


#1. 데이타 
path = './_data/dacon_shopping/'                                         
train_set = pd.read_csv(path + 'train.csv', index_col=0)                              

print(train_set)
print(train_set.shape)   # (6255, 12)
print(train_set.columns)
# ['Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1', 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment', 'IsHoliday', 'Weekly_Sales']

test_set = pd.read_csv(path + 'test.csv', index_col=0)    
print(test_set.columns)
# ['Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1', 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment', 'IsHoliday'] !!! Weekly_Sales !!!

print(test_set.shape)    # (180, 11)




print(train_set.isnull().sum()) 
print(test_set.isnull().sum()) 
# print(train_set.describe()) 


"""
Store              0
Date               0
Temperature        0
Fuel_Price         0
Promotion1      4153
Promotion2      4663
Promotion3      4370
Promotion4      4436
Promotion5      4140
Unemployment       0
IsHoliday          0
Weekly_Sales       0
"""
train_set = train_set.dropna()
test_set = test_set.fillna(test_set.mean())
print(train_set.isnull().sum())
print(train_set.shape)
print(test_set.shape)


# date를 month로 문자를 숫자로 변환  "월"은 "31/08/2012" 형태의 값 중 4번째~5번째 글자
# Date에서 월을 Month 컬럼으로 분리시킨다 그리고 Date컬럼을 없앤다

def get_month(date):
    month = date[3:5]
    month = int(month)
    return month

train_set['Month'] = train_set['Date'].apply(get_month)
test_set['Month'] = test_set['Date'].apply(get_month)

train_set = train_set.drop(['Date'], axis=1)
test_set = test_set.drop(['Date'], axis=1)

# boolean값이 단일 열 일때 0 or 1로 변환
train_set["IsHoliday"] = train_set["IsHoliday"].astype(int)
test_set["IsHoliday"] = test_set["IsHoliday"].astype(int)
"""
train_set['Promotion1'] = train_set['Promotion1'].fillna(train_set.Promotion1.dropna().mode()[0])  # NaN값을 최빈값으로 채운다
test_set['Promotion1'] = test_set['Promotion1'].fillna(train_set.Promotion1.dropna().mode()[0])
train_set['Promotion2'] = train_set['Promotion2'].fillna(train_set.Promotion2.dropna().mode()[0])  # NaN값을 최빈값으로 채운다
test_set['Promotion2'] = test_set['Promotion2'].fillna(train_set.Promotion2.dropna().mode()[0])
train_set['Promotion3'] = train_set['Promotion3'].fillna(train_set.Promotion3.dropna().mode()[0])  # NaN값을 최빈값으로 채운다
test_set['Promotion3'] = test_set['Promotion3'].fillna(train_set.Promotion3.dropna().mode()[0])
train_set['Promotion4'] = train_set['Promotion4'].fillna(train_set.Promotion4.dropna().mode()[0])  # NaN값을 최빈값으로 채운다
test_set['Promotion4'] = test_set['Promotion4'].fillna(train_set.Promotion4.dropna().mode()[0])
train_set['Promotion5'] = train_set['Promotion5'].fillna(train_set.Promotion5.dropna().mode()[0])  # NaN값을 최빈값으로 채운다
test_set['Promotion5'] = test_set['Promotion5'].fillna(train_set.Promotion5.dropna().mode()[0])
"""

train_set['Promotion1'] = train_set['Promotion1'].fillna(train_set.mean())  
test_set['Promotion1'] = test_set['Promotion1'].fillna(test_set.mean())  
train_set['Promotion2'] = train_set['Promotion2'].fillna(train_set.mean())  
test_set['Promotion2'] = test_set['Promotion2'].fillna(test_set.mean())  
train_set['Promotion3'] = train_set['Promotion3'].fillna(train_set.mean())  
test_set['Promotion3'] = test_set['Promotion3'].fillna(test_set.mean())  
train_set['Promotion4'] = train_set['Promotion4'].fillna(train_set.mean())  
test_set['Promotion4'] = test_set['Promotion4'].fillna(test_set.mean())  
train_set['Promotion5'] = train_set['Promotion5'].fillna(train_set.mean())  
test_set['Promotion5'] = test_set['Promotion5'].fillna(test_set.mean())  






print(train_set)





x = train_set.drop(['Weekly_Sales'], axis=1)
print(x.columns)
# ['Store', 'Date', 'Temperature', 'Fuel_Price', 'Promotion1', 'Promotion2', 'Promotion3', 'Promotion4', 'Promotion5', 'Unemployment', 'IsHoliday']

y = train_set['Weekly_Sales']
print(y)
print(y.shape)

print(train_set)
print(test_set)



x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=3
)


# scaler = MinMaxScaler() 
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)                      # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)        # train이 변한 범위에 맞춰서 변환됨
test_set = scaler.transform(test_set)  
# y_summit = model.predict(test_set) test셋은 스케일링이 상태가 아니니 summit전에 스케일링을 해서  y_summit = model.predict(test_set) 에 넣어줘야 한다 
# summit하기 전에만 해주면 상관이 없다

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위



#2. 모델구성
input_01 = Input(shape=(11,))
dense_01 = Dense(100)(input_01)
dropout_01 = Dropout(0.2)(dense_01)
dense_02 = Dense(100, activation="relu")(dropout_01)
dropout_02 = Dropout(0.3)(dense_02)
dense_03 = Dense(100, activation="relu")(dropout_02)
dropout_03 = Dropout(0.5)(dense_03)
dense_04 = Dense(100, activation="relu")(dropout_03)
dropout_04 = Dropout(0.2)(dense_04)
dense_05 = Dense(100, activation="relu")(dropout_04)
output_01 = Dense(1)(dense_05)
model = Model(inputs=input_01, outputs=output_01)
model.summary()





#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)     

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))  




rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           




y_summit = model.predict(test_set)
# y_summit = abs(y_summit)
print(y_summit)
print(y_summit.shape) # (715, 1)




sampleSubmission = pd.read_csv('./_data/dacon_shopping/sample_submission.csv')
sampleSubmission['Weekly_Sales'] = y_summit
print(sampleSubmission)
sampleSubmission.to_csv('./_data/dacon_shopping/sample_submission_m.csv', index = False)


 

# scaler = MinMaxScaler() 
# r2스코어 : 0.19579441848448165
# RMSE : 518612.80763140286

# scaler = StandardScaler()
# loss : [388003.0, 241570152448.0]
# r2스코어 : 0.27768958593093807
# RMSE : 491497.8638255445

# scaler = RobustScaler()
# loss : [394341.65625, 252936208384.0]
# r2스코어 : 0.24370438532162153
# RMSE : 502927.6205056404


# scaler = MaxAbsScaler()
# loss : [275162234880.0, 431448.09375]
# r2스코어 : 0.17724708774813436
# RMSE : 524559.0818549548

# 데이콘 따릉이 문제풀이

import datetime as dt
import numpy as np                                               
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error         

#1. 데이타 
path = './_data/kaggle_bike/'      
                                  
train_set = pd.read_csv(path + 'train.csv', index_col=0)                       
print(train_set)
print(train_set.shape) #(10886, 11) 컬럼 11개



test_set = pd.read_csv(path + 'test.csv', index_col=0)                                   
print(test_set)        
print(test_set.shape) 



print(train_set.columns)     # (6493, 8)
print(train_set.info())      
print(train_set.describe()) 








#### 결측치 처리 ####
"""
# object -> 날짜 형식으로 변환
train_set['datetime'] = pd.to_datetime(train_set['datetime'])
test_set['datetime'] = pd.to_datetime(test_set['datetime'])

train_set['year'] = train_set['datetime'].dt.year
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour
train_set['minute'] = train_set['datetime'].dt.minute
train_set['second'] = train_set['datetime'].dt.second
# 요일 (0: 월 ~ 6: 일)
train_set['dayofweek'] = train_set['datetime'].dt.dayofweek
"""






print(train_set.isnull().sum()) 
test_set = test_set.fillna(test_set.mean())  # 결측지처리 nan 값에 0 기입   추가코드
train_set = train_set.dropna()  
print(train_set.isnull().sum())
print(train_set.shape)     





x = train_set.drop(['casual', 'registered', 'count'], axis=1)   #drop 뺀다         axis=1 열이라는걸 명시
print(x)
print(x.columns) #[10886 rows x 8 columns]
print(x.shape)   #(10886, 8)   input_dim=8

y = train_set['count']   #카운트 컬럼만 빼서 y출력---------- (sampleSubmission.csv 에서 구하려고 하는값이 count값이라서?? )
print(y)  
print(y.shape)  #(10886,) 10886개의 스칼라  output 개수 1개       여기까지 #1 데이터 부분을 잡은것

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.99,
    shuffle=True,     # 12 = 124  15까지 했음
    random_state=5
    )


#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=8))      
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')               
model.fit(x_train, y_train, epochs=1......................000, batch_size=50)  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)     

y_predict = model.predict(x_test) 

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))  




rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           




y_summit = model.predict(test_set)
y_summit = abs(y_summit)
print(y_summit)
print(y_summit.shape) # (715, 1)




sampleSubmission = pd.read_csv('./_data/kaggle_bike/sampleSubmission.csv')
sampleSubmission['count'] = y_summit
print(sampleSubmission)
sampleSubmission.to_csv('./_data/kaggle_bike/sampleSubmission_m.csv', index = False)


# loss : 104.90857696533203
# RMSE : 146.90824146658113
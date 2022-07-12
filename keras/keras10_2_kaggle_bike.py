from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error          # mean_squared_error : RMSE
import numpy as np                                               
import pandas as pd



#1. 데이타 
path = './_data/kaggle_bike/'      
                                  
train_set = pd.read_csv(path + 'train.csv', index_col=0)   
                    
print(train_set)
print(train_set.columns)   # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
print(train_set.shape)     # (10886, 11) 컬럼 11개


test_set = pd.read_csv(path + 'test.csv', index_col=0)  

print(test_set)                                  
print(test_set.columns)    # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']  
print(test_set.shape)      # (6493, 8)


print(train_set.isnull().sum())   
"""
 season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0
dtype: int64
"""

print(test_set.isnull().sum())    
"""
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
dtype: int64
"""
print(train_set.describe()) 
print(train_set.shape)   # (10886, 11)
print(test_set.shape)    # (6493, 8)   'casual', 'registered', 'count'




#### 결측치 처리 ####
x = train_set.drop(['casual', 'registered', 'count'], axis=1)  
print(x)
print(x.columns)         # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
print(x.shape)           # (10886, 8)

y = train_set['count']   
print(y)  
print(y.shape)           # (10886,) 10886개의 스칼라 output=1    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=5)  
    


#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))      
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
model.fit(x_train, y_train, epochs=300, batch_size=100)  



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)     

y_predict = model.predict(x_test) 

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))  

rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           



# y_summit = model.predict(test_set)
# y_summit = abs(y_summit)
# print(y_summit)
# print(y_summit.shape) # (715, 1)

# sampleSubmission = pd.read_csv('./_data/kaggle_bike/sampleSubmission.csv')
# sampleSubmission['count'] = y_summit
# print(sampleSubmission)
# sampleSubmission.to_csv('./_data/kaggle_bike/sampleSubmission_m.csv', index = False)


# loss : 114.30289459228516
# RMSE : 163.08621823284489
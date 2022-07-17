# 평가까지
# lstm Conv1D  
# 스플릿트 사용(컬럼이 뭉치로 사용되어야 한다)

import numpy as np
from sklearn import metrics          
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D 
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score, mean_squared_error  
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
  
train_set = pd.read_csv('./_data/kaggle_jena/jena_climate_2009_2016.csv')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os

path = './_data/kaggle_jena/'
data = pd.read_csv(path + 'jena_climate_2009_2016.csv' )


data['T (degC)'].plot(figsize=(12,6)) 
# plt.show() # 'T (degC)'열의 전체 데이터 시각화

plt.figure(figsize=(20,10),dpi=120)
plt.plot(data['T (degC)'][0:6*24*365],color="black",linewidth=0.2)

print(data) #(420551, 15)
print(data.info()) #(420551, 15)
data.index = pd.to_datetime(data['Date Time'],
                            format = "%d.%m.%Y %H:%M:%S") 

print(data.info()) 

hourly = data[5::6] 

print(hourly) #(70091, 15)



hourly = hourly.drop_duplicates()
print(hourly) #[70067 rows x 15 columns] 
hourly.duplicated().sum() 

daily = data['T (degC)'].resample('1D').mean().interpolate('linear')

daily[0:365].plot()
# plt.show() # 월별 온도 확인

hourly_temp = hourly['T (degC)']
len(hourly_temp) 
print(len(hourly_temp)) #70067

def generator(data, window, offset):
    gen = data.to_numpy() #데이터 프레임을 배열객체로 반환
    X = []
    y = []
    for i in range(len(gen)-window-offset): # 420522
        row = [[a] for a in gen[i:i+window]] #행
        X.append(row)
        label = gen[i+window+offset-1]
        y.append(label)
    return np.array(X), np.array(y)
WINDOW = 5
OFFSET = 24

X, y = generator(hourly_temp, WINDOW, OFFSET)
# print(X,X.shape) #(70038, 5, 1)
# print(y,y.shape) #(70038,)
gen = data.to_numpy()
label = gen[0+WINDOW+OFFSET-1] 

print(label,label.shape)# (15,)


X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]
print(X_train,y_train)
print(X_train.shape,y_train.shape) #(60000, 5, 1) (60000,)



from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import InputLayer
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


#2. 모델구성
model1 = Sequential()
model1.add(InputLayer((WINDOW, 1)))
model1.add(LSTM(100))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()

#3. 컴파일,훈련
model1.compile(loss='mae', optimizer='Adam')
model1.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

#4. 평가,예측
loss = model1.evaluate(X_test, y_test)
print("loss :",loss)
test_predictions = model1.predict(X_test).flatten()

result = pd.DataFrame(data={'Predicted': test_predictions, 'Real':y_test})
plt.figure(figsize=(20,7.5),dpi=120)
plt.plot(result['Predicted'][:300], "-g", label="Predicted")
plt.plot(result['Real'][:300], "-r", label="Real")
plt.legend(loc='best')
result['Predicted'] = result['Predicted'].shift(-OFFSET)
result.drop(result.tail(OFFSET).index,inplace = True)
print(result)

# loss : 2.4077389240264893




from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes     
import numpy as np 
# [실습]
# R2 0.62 이상

#1. 데이터                        
datasets = load_diabetes()                      
x = datasets.data
y = datasets.target 

print(x) 
print(y) 
print(x.shape, y.shape)         #(442, 10) (442,)     컬럼 = 10    스칼라 = 422

print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR) 

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9,
    shuffle=False
    )



#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                   
model.fit(x_train, y_train, epochs=500, batch_size=10)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# R2 0.62 이상
"""
loss : 1773.4490966796875
r2스코어 : 0.6805337042324608

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))

model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                   
model.fit(x_train, y_train, epochs=500, batch_size=1000)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
"""
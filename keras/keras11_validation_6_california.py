from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense  
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.datasets import fetch_california_housing 



#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target                                  

print(x)
print(y)
print(x.shape, y.shape)        # (20640, 8) (20640,)    
print(datasets.feature_names)  # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)          # Number of Instances: 20640(행) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)



#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=8))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                   
model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split=0.2)



#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)


"""
loss : 0.6425122618675232 ----- Normal
r2 : 0.5317547523397925

loss : 0.6713584661483765 ----- validation_split
r2 : 0.5107321081546246
"""
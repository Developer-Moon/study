from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes     
import numpy as np

#1. 데이터                        
datasets = load_diabetes()                      
x = datasets.data
y = datasets.target 
print('y의 라벨값 :', np.unique(y, return_counts=True))
print('y의 라벨값 :', np.unique(x, return_counts=True))
print(x.shape, y.shape)         # (442, 10) (442,)     컬럼 = 10 스칼라 = 422
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=72)



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
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss :', loss)
print('r2스코어 :', r2)

# loss : 2189.79541015625
# r2스코어 : 0.634477819866959
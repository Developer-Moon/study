from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split 
import numpy as np  
from sklearn.datasets import load_boston   


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      

print(x)                        
print(y)                  
print(x.shape, y.shape)         # (506, 13) (506,) - 데이터 : 506개,   컬럼 : 13 - input_dim (506개의 스칼라 1개의 벡터)
print(datasets.feature_names)   # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] 컬럼셋 b는 흑인이라 사용X
print(datasets.DESCR)  


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=13))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                   
model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2) 



#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)      

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)


"""
loss : 21.61882972717285 ----- Normal
r2 : 0.738325009535564

loss : 20.06865882873535 ----- validation_split
r2 : 0.7570883292496796
"""



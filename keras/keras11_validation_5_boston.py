from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense  
import numpy as np 
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      

print(x)                  
print(y)                  

print(x.shape, y.shape)         #(506, 13) (506,) 506개의 데이터 개수   13개의 컬럼 (input_dim=13)      (506개의 스칼라 1개의 벡터)

print(datasets.feature_names)   # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] 컬럼셋 b는 흑인 그래서 이 데이터 셋은 못 쓰게한다

print(datasets.DESCR)           #DESCR 설명하다 묘사하다 - 컬럼들의 소개가 나온다


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7,
    random_state=3
    )



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13))
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
model.compile(loss='mse', optimizer='adam')                   
model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.25) #, validation_split=0.25


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
validation = model.evaluate()
print('loss :', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


####################################
# loss : 31.8258056640625        o
# r2스코어 : 0.5953513272342723 

# loss : 36.62041091918945
# val_loss: 33.1409    
# r2스코어 : 0.5343903984976897   X
####################################




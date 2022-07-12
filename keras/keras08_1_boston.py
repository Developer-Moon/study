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
print(datasets.DESCR)           # 컬럼들의 소개가 나온다

"""
:Number of Instances: 506                                                                                    (행)

:Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target. (열)

:Attribute Information (in order):                                                                           13개의 상세내용
    - CRIM     per capita crime rate by town
    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS    proportion of non-retail business acres per town
    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX      nitric oxides concentration (parts per 10 million)
    - RM       average number of rooms per dwelling
    - AGE      proportion of owner-occupied units built prior to 1940
    - DIS      weighted distances to five Boston employment centres
    - RAD      index of accessibility to radial highways
    - TAX      full-value property-tax rate per $10,000
    - PTRATIO  pupil-teacher ratio by town
    - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
    - LSTAT    % lower status of the population
    - MEDV     Median value of owner-occupied homes in $1000's

:Missing Attribute Values: None
"""

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
model.fit(x_train, y_train, epochs=300, batch_size=10)



#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)      # 예측값

from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)

# 2. R2 0.8 이상


# loss : 21.61882972717285
# r2 : 0.738325009535564


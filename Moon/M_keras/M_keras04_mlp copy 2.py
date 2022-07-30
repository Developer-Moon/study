from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense  
import numpy as np

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],                    
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]])
y = np.array([11,12,13,14,15,16,17,18,19,20])

x = x.T



#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=2))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일, 훈련
model.
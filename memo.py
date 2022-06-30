from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import numpy as np 

#1. 데이터
x = np.array([1,2,3])                              # numpy에서는 기본적으로 array(배열, 행렬)라는 단위로 데이터를 관리하며 이에 대해 연산을 수행
y = np.array([1,2,3])


#2. 모델구성
model = Sequential()  

model.add(Dense(4, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일(compile =컴퓨터가 알아듣게), 훈련
model.complie(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([4])
print(result)





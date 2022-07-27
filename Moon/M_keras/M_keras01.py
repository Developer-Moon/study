from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])



#2. 모델구성
model =Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300)



#4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([4])

print('loss : ', loss)
print('4의 예측값 : ', result)

# loss :  0.004123168531805277
# 4의 예측값 :  [[3.871759]]
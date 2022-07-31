from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])



#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=350)



#4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([6])

print(loss)
print(result)
# 0.3800000250339508
# [[5.699342]]


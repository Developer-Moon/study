from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
import numpy as np  

#1. 데이터 
# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])             # test셋은 evaluate, predict에서 사용
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])              # val 검증의 약자 = validation(검증)             필기 중 : 수능보러가기 단계

x = np.array(range(1, 17))
y = np.array(range(1, 17))


# [실습] 슬라이싱으로 잘라봐라
x_train = x[:9]
x_val = x[9:13]
x_test = x[13:17]

y_train = y[:9]
y_val = y[9:13]
y_test = y[13:17]




#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # 훈련하고 문제풀고 훈련하고 문제풀고


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                                 # 여기서 수능보고 
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)


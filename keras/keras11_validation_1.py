from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
import numpy as np  

#1. 데이터 
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])             # test셋은 evaluate, predict에서 사용
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val = np.array([14,15,16])              # val 검증의 약자 = validation(검증)             필기 중 : 수능보러가기 단계


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

"""
Epoch 100/100
10/10 [==============================] - 0s 2ms/step - loss: 0.0668 - val_loss: 0.4221     <----    여기에 validation이 생긴다  검증을 했을때 로스보다 값이 더 좋지 않아야한다 
1/1 [==============================] - 0s 67ms/step - loss: 0.1695                                  과적합 구간인 경우 로스가 더 좋지 않을 수 있다
loss :  0.16945324838161469
17의 예측값 :  [[16.193598]]
"""
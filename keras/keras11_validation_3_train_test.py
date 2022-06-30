from sklearn.model_selection import train_test_split
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



#train, test 6:4로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size = 0.65, 
                                                    random_state = 42)
print(x_train)
#test set을 5:5로 test, val 나누기
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
                                                 test_size=0.5,
                                                 random_state = 42)





print(x_train)
print(x_test)
print(x_val)
"""
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

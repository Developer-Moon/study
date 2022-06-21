import numpy as np   
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense    


#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 
# x_train = np.array([1, 2, 3, 4, 5, 6, 7])
# x_test = np.array([8, 9, 10])
# y_train = np.array([1, 2, 3, 4, 5, 6, 7])
# y_test = np.array([8, 9, 10])


#과제 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라

x_train = x[0:7]
x_test = x[0:7]
y_train = x[7:10]
y_test = x[7:10]

print(x_train)
print(y_test)
print(x_train)
print(y_test)





#2. 모델구성
model = Sequential()  
model.add(Dense(10, input_dim=1))          #히든레이어가 1개라고 딥러닝이 아니라는거 아니다 딥러닝이다  레이어가 없어도 성능만 좋으면 굿
model.add(Dense(1))
 
 
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)
 
 
 #4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss :', loss)
result = model.predict([11]) 
print('11의 예측값은 :', result)

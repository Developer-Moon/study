from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
import numpy as np  

# train_test_split으로만 10 : 3 : 3 으로 나눠라 

#1. 데이터 
x = np.array(range(1, 17))
y = np.array(range(1, 17))


#train, test 6:4로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.65, random_state = 42)
print(x_train) # [ 9 10  3 16  5  8 11 13  4  7]
print(x_test)  # [ 1  2  6 15 14 12]


#test set을 5:5로 test, val 나누기
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state = 42)
print(x_train) # [ 9 10  3 16  5  8 11 13  4  7]
print(x_test)  # [ 6 14 15]
print(x_val)   # [ 1  2 12]



"""
#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
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

import numpy as np   
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense    


#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 


#과제 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라                   [이런식으로 절대 안 자른다 다음 페이지에서 설명]
x_train = x[0:7]    # = x[:7]                                  30%를 자른다면 [1, 2, 4, 5, 7, 8, 9] 중간에 랜덤하게 뺀다 (셔플) 하는 이유 -  
x_test = x[0:7]     # = x[7:]                                  이런식으로 작업하면 일부를 변화 시킬때 나눈지점에서 기울기가 틀어진다 그러지 않기 위해 셔플하여 자른다?
y_train = x[7:10]
y_test = x[7:10]

print(x_train)      # [1 2 3 4 5 6 7]
print(y_test)       # [ 8  9 10]
print(x_train)      # [1 2 3 4 5 6 7]
print(y_test)       # [ 8  9 10]


#2. 모델구성
model = Sequential()  
model.add(Dense(10, input_dim=1))      
model.add(Dense(1))
 
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)
 
 
 #4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss :', loss)
result = model.predict([11]) 
print('11의 예측값은 :', result)

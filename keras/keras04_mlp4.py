import numpy as np 
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 

#1. 데이터
x = np.array([range(10)])    
#print(range(10))
#for i in range(10):          
#    print(i)
print(x.shape) #(1, 10)
x = np.transpose(x)                 
print(x.shape) #(10, 1)


y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
             [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])           # 3행 10열 이므로 트랜스포즈
y = np.transpose(y)                                     # y를transpose한다음에 덮어쓰겠다
print(y.shape) #(10, 3)


#2 모델
#[실습] 맹그러봐      
model = Sequential()
model.add(Dense (5, input_dim=1)) 
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (3))



#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=1) 



#4 결과, 예측 [[9]]   10, 1.9 0 얼마나 가깝고 loss가 얼마나 줄어드느냐
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([[9]])
print('[9]의 예측값 : ', result)

# loss : 2.009651532642806e-13
# [9]의 예측값 :  [[ 1.0000000e+01  1.8999999e+00 -4.1723251e-07]]
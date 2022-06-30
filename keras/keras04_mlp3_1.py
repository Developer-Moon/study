import numpy as np 
from tensorflow.keras.models import Sequential      
from tensorflow.keras.layers import Dense 

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])        # range 함수는 연속적인 숫자 객체를 만들어서 반환해주는 함수 - range(10) 호출 : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9가 반환     
                                                                 # range의 최소값은 0, 음수의 숫자를 다루지 못한다
#print(range(10))
#for i in range(10):                                             # for문:반복문  해석 : 0부터 9까지 i라는 인수에 반복 해라
#    print(i)
print(x.shape)      # (3, 10)

x = np.transpose(x) # 행렬 변환                 
print(x.shape)      # (10, 3)


y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],          
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])  # x.shape는 (3,10) y,shape는 (2,10) 이므로 행렬 변환(transpose)
y = np.transpose(y)                                              # y를 변환(transpose)한 다음 덮어쓰겠다
print(y.shape)


#[실습] 맹그러봐  
#2 모델  
model = Sequential()
model.add(Dense (5, input_dim=3)) 
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (2))



#3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1) 



#4 결과, 예측 [[9, 30, 210]]   10, 1.9 얼마나 가깝고 loss가 얼마나 줄어드느냐
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([[9, 30, 210]])
print('[9, 30, 210]의 예측값 : ', result)


# loss : 4.082849683340051e-11
# [9, 30, 210]의 예측값 :  [[10.000007   1.8999887]]
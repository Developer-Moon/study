import numpy as np 
from tensorflow.keras.models import Sequential      
from tensorflow.keras.layers import Dense     

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],                    
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4]]
             )
y = np.array([11,12,13,14,15,16,17,18,19,20])   


print(x.shape) # (2, 10) 이걸 (10,2)로 바꿔줘야한다        
print(y.shape) # (10,) -> (10,1)

""" 행과열을 바꾸는 첫번째 방법"""
x = x.T # 행과 열을 바꿔(전치 해)준다 - 연산해서 다시 x에 넣어준다
print(x)
print(x.shape)


""" 행과열을 바꾸는 두번째 방법
x = x.transpose()
print(x)
print(x.shape)
"""

"""
# 와꾸만 바뀌고 순서는 안 바뀐다 - but 많이 쓴다
x = x.reshape(10,2)
print(x)
print(x.shape)
"""

#2. 모델구성
model = Sequential()
model.add(Dense (5, input_dim=2))  #(10,2) 디멘션이 2 = 열의 개수(피처, 특성, 컬럼의 개수)                 기존코드 - model.add(Dense (5, input_dim=1)) 
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (4))
model.add(Dense (3))
model.add(Dense (2))
model.add(Dense (1))


#3. 컴파일, 훈련0
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=1)        # batch_size=3일때 123, 456, 789, 10 으로 훈련시킨다


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)
result = model.predict([[10, 1.4]]) # 원래 (2, )인데  [[-]] 괄호가 2개 들어가서 (1,2) 가 된다 ------------------ 열 우선인 이유 
print('[10, 1.4]의 예측값 : ', result)            # ValueError : Data cardinality is ambiguous:   값에 대한 에러가있다 : x사이즈는 2개 y사이즈는 10개다 = x값의 모양과 y값의 모양이 다르다
#[10, 1.4]의 예측값 :  [[20.013554]]              # x sizes: 2 y sizes: 10

print(x)
print(x.shape) #(10, 2)
# sdsdsd 123123    33333333
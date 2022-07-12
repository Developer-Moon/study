from tensorflow.python.keras.models import Sequential    
from tensorflow.python.keras.layers import Dense 
import numpy as np    



# [실습] 맹그러봐!! [6]을 예측한다
#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])



#2. 모델구성
model = Sequential()
model.add(Dense(2, input_dim=1)) 
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') # loss = mse, mae 등등
model.fit(x, y, epochs=1000)                # 훈련량에 따라 성능은 알 수 없지만 가성비는 훈련량이 낮을수록 좋다  

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])  
print('예측값은 : ', result)

# loss :  0.4005308151245117
# 예측값은 :  [[5.990535]]
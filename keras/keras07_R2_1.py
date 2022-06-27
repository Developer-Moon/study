from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  
import numpy as np 
from sklearn.model_selection import train_test_split  

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7,
    shuffle=False,
    # random_state=66
    )


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                   
model.fit(x_train, y_train, epochs=200, batch_size=1)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x) 

from sklearn.metrics import r2_score       
r2 = r2_score(y, y_predict)                             #결과를 스코어로 반환
print('r2스코어 :', r2) 



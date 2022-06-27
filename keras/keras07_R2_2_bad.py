#1. R2를 음수가 아닌  0.5 이하로 만들것 - 강제적으로 멍청한 모델을 만들어라
#2. 데이터 건들지 마라
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. train 70%
#7. epochs 100번 이상
#8. [실습시작]




from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  
import numpy as np 
from sklearn.model_selection import train_test_split  

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7,
    shuffle=True,
    random_state=66
    )


#2. 모델구성
model = Sequential()
model.add(Dense(55, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))

model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')                   
model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x) 

from sklearn.metrics import r2_score      
r2 = r2_score(y, y_predict)                
print('r2스코어 :', r2)


# loss : 3.1823184490203857
# r2스코어 : 0.16396925864897582
# 작업한부분 : 모델구성 node의 개수와 레이어 개수 수정
 
 



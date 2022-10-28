from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense   
import numpy as np 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

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
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')              # 회귀모델 : 결과값이 딱 떨어지는게 아이라...  이 모델의 보조는 R2(결정계수)모델 or R제곱   R2의 수치가 높을수록 좋다
model.fit(x_train, y_train, epochs=500, batch_size=1)    # R2는 보조지표?                             에큐러시에서 1과 0.00001은 다르다



#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x)                             # weight가 수정된 값
 
          
plt.scatter(x, y)                                        # 뿌리다 점을 찍겠다
plt.plot(x, y_predict, color='red')                      # x와 y를 잇는 선을 그리다 - 빨강선이 나온다 (파이썬은 css처럼 바로 적용시키나?)
plt.show()                                               # 맵이 뜬다  그리기      평가 지표는 항상 2개 이상 잡는다



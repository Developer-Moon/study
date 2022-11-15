from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 
import numpy as np 
from sklearn.model_selection import train_test_split 

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9, 
    random_state=66
    )




#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1000) 



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x)

from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)          # 결과를 스코어로 반환

print('r2스코어 : ', r2)
# loss : 0.8285454511642456
# r2스코어 :  0.7325447926415286



# import matplotlib.pyplot as plt              


# plt.scatter(x, y)
# plt.plot(x, y_predict, color='red')   
# plt.show()





# 에큐러시라는 지표로 확률을 구하면 다른 로스가 0.0001이 나와도 확률은 0이다
# 0.0001정도 수치를 ~~어째 하는게 r2지표  결정계수 r2스코어???
# 로스 값이 낮아도 r2값이 낮을 수 있다 
# ex)  로스 0.001 r2 98
#      로스 0.0001 r2 97 일때       통상 로스가 낮은걸 신뢰한다






"""
결정계 수란 = '회귀 모델의 성과 지표'
1에 가까울 수록 좋은 회귀 모델
0에 가까울 수록 나쁜 모델
음수가 나올경우, 바로 폐기해야 하는 모델

그렇다면 음수가 나오는 이유는??
"""
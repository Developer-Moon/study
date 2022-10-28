import numpy as np      
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from sklearn.model_selection import train_test_split   

# LSTM, GRU 두 가지의 모델 중 성능은 비슷하며
# LSTN이 데이터가 더 많을때 정확도가 더 높다는 논문이 많다

#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8,], [7, 8, 9]]) 
#              timesteps 3개씩 자르겠다
y = np.array([4, 5, 6, 7, 8, 9, 10])


print(x.shape, y.shape) # (7, 3) (7,)
x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)



#2. 모델구성
model = Sequential()  
model.add(GRU(units=10, activation='relu', input_shape=(3,1))) 
model.add(Dense(20))
model.add(Dense(1))
model.summary()

"""
3 * (다음 노드 수^2 +  다음 노드 수 * Shape 의 feature + 다음 노드수 )
3   (        100x100         100x1                         100)     = 30600    




[GRU]      units : 10 -> 3 x 10 x (1 + 1 +10) = 300

결론 : GRU = simpleRNN x 3

"""



    
    
    

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x, y, epochs=700, batch_size=32) #  validation_split=0.2



#4. 결과, 예측
loss = model.evaluate(x, y) # 2차원이라 3차원으로 변경해야한다
# 기존의 np.array([8,9,10])는 2차원이다 (3, ) - (3, 1)
y_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[[8], [9], [10]]]
result = model.predict(y_pred)


print("loss :", loss)
print("[8,9,10]의 예측 결과", result)

# epochs=700
# loss : 4.106235792278312e-05
# [8,9,10]의 예측 결과 [[11.00104]]
import numpy as np      
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout
from sklearn.model_selection import train_test_split   

#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8,], [7, 8, 9]]) 
y = np.array([4, 5, 6, 7, 8, 9, 10])


print(x.shape, y.shape) # (7, 3) (7,)
# x의 shape = (행, 열, 몇개씩 짜르는지!!!) timesteps
# RNN은 shape가 3차원 - (N, 3, 1) 여기서 3은 자르는 단위 

x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)



#2. 모델구성
model = Sequential()  
model.add(SimpleRNN(13, input_shape=(3, 4)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
model.summary()




"""
# model = Sequential()  
# model.add(SimpleRNN(units=10, input_shape=(3, 4))) 
# model.add(Dense(5, activation='relu'))
# model.add(Dense(1))
# model.summary()


# param 공식
# param = 파라미터 아웃값 X (파라미터 아웃값 + 디멘션 값 + 1(bais))
# param = num_units      X (num_units + input_dim + 1)
              10    x (    10    +     4     + 1) = 221
                                

         #  units   x (   units  +   맨 뒷자리 + 1 ) 여기서 1은 bias
         #                           맨 뒷자리 - input_shape=(3, 4) 여기서 4 
                
         
"""






"""
model = Sequential()    # (하단 input_shape=(3, 1)) 행무시 : 7 무시
model.add(SimpleRNN(10, input_shape=(3, 1))) # RNN은 2차원으로 전달해서 Flatten없이 바로 Dense 사용 가능
model.add(SimpleRNN(10))
#  ValueError: Input 0 of layer simple_rnn_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 10)
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
"""
"""
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x, y, epochs=180, batch_size=1,) #  validation_split=0.2



#4. 결과, 예측
loss = model.evaluate(x, y) # 2차원이라 3차원으로 변경해야한다
# 기존의 np.array([8,9,10])는 2차원이다 (3, ) - (3, 1)
y_pred = np.array([8,9,10]).reshape(1, 3, 1) # [[[8], [9], [10]]]
result = model.predict(y_pred)


print("loss :", loss)
print("[8,9,10]의 예측 결과", result)
"""
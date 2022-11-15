import numpy as np      
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM 
from sklearn.model_selection import train_test_split   

# LSTM은 시간이 느리다!!

#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8,], [7, 8, 9]]) 
#              timesteps 3개씩 자르겠다
y = np.array([4, 5, 6, 7, 8, 9, 10])


print(x.shape, y.shape) # (7, 3) (7,)
# x의 shape = (행, 열, 몇개씩 짜르는지!!!) timesteps
# RNN은 shape가 3차원 - (N, 3, 1) 여기서 3은 자르는 단위 

x = x.reshape(7, 3, 1)
print(x.shape) # (7, 3, 1)



#2. 모델구성
model = Sequential()  
# model.add(SimpleRNN(10, input_shape=(3, 1))) # [batch, timesteps, feature].
model.add(LSTM(units=100, activation='relu', input_shape=(3,1))) # input_shape가  input_length=3, input_dim=1 로 사용가능
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.summary()


# [LSTM] units : 10 - 4 * 1- * (1 + 1 +10) = 400
#        units : 4 * 20 * (1 + 1 +20) = 1760
       
# 결론 : LSTM = simpleRNN * 4
# 숫자4의 의미는 cell state, input gate, output agte, forget gate


# model.add(LSTM(units=8, input_length=3, input_dim=2))

"""
Params = 4*((input_shape_size +1) * ouput_node + output_node^2)
params = 4*((1+1) * 8 + 8^2) 가 되어 320 값이 나오게됩니다.
"""

# 4를 곱하는 이유는 LST에 들어있는 4개의 상호작용하는 레이어가 있는 반복되는 모듈이 있어서다
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
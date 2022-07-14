from matplotlib import units
import numpy as np      
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten, Conv1D # Conv1D 3차원

from sklearn.model_selection import train_test_split   



#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8,], [7, 8, 9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) # (7, 3) (7,)
# x의 shape = (행, 열, 몇개씩 짜르는지!!!) 
# RNN은 shape가 3차원 - (N, 3, 1) 여기서 3은 자르는 단위 

x = x.reshape(7, 3, 1)



#2. 모델구성
model = Sequential()    
# model.add(LSTM(10, input_shape=(3,1), return_sequences=False))
model.add(Conv1D(10, 2, input_shape=(3, 1))) 
model.add(Flatten())
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
model.summary()


# Total params: 517
# Trainable params: 517
# Non-trainable params: 0
# _______________________

# Total params: 97
# Trainable params: 97
# Non-trainable params: 0


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x, y, epochs=180, batch_size=1,)



#4. 결과, 예측
loss = model.evaluate(x, y)                   
y_pred = np.array([8,9,10]).reshape(1, 3, 1)   # [[[8], [9], [10]]]  -  기존np.array([8,9,10]) 2차원(3, )이라 3차원으로 변경해야한다
result = model.predict(y_pred)

print("loss :", loss)
print("[8,9,10]의 예측 결과", result)




# DNN - (f + b) x unit  
# REN - (f + b +unit) x units
# Bidirectional - ((f + b +unit) x units) x 2
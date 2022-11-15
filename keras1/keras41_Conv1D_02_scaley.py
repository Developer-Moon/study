import numpy as np   
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.models import Sequential  
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D #Conv1D 3차원



#1. 데이터 
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10], [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = np.array([50, 60, 70])
print(x_predict.shape) #(3,)
print(x.shape, y.shape) # (13, 3) (13,)
x = x.reshape(13, 3, 1)

# 돈 날씨 같은 경우 LSTM을 사용

#2. 모델구성
model = Sequential()  
model.add(Conv1D(30, 2, input_shape=(3, 1)))
model.add(MaxPooling1D(1,1))   
model.add(Dropout(0.3))
model.add(Conv1D(10, 1, padding='valid', activation='relu'))
model.add(MaxPooling1D(1, 1))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.summary()



#3. 컴파일, 훈련
from tensorflow.python.keras.callbacks import EarlyStopping  
model.compile(loss='mse', optimizer='adam') 
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, restore_best_weights=True)     
model.fit(x, y, epochs=300, batch_size=2, callbacks=[earlyStopping]) 



#4. 결과, 예측
loss = model.evaluate(x, y)
y_pred = x_predict.reshape(1, 3, 1) # [[[8], [9], [10]]]
result = model.predict(y_pred)


print("loss :", loss)
print("x_predict의 예측 결과", result)


# 유워너 80? ㅇㅋ

# loss : 0.007141932379454374
# x_predict의 예측 결과 [[80.63769]]


# GRU
# loss : 0.013739822432398796
# x_predict의 예측 결과 [[74.695984]]


# LSTM
# loss : 0.06442474573850632
# x_predict의 예측 결과 [[74.10001]]

# SimpleRNN
# loss : 0.00036197842564433813
# x_predict의 예측 결과 [[74.10104]]

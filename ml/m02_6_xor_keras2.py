import numpy as np
from sklearn.svm import LinearSVC  
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense

# 퍼셉트론을 텐서로 표현

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] # XOR gate 같으면 0 틀리면 1
y_data = [0, 1, 1, 0]



#2. 모델
# model = LinearSVC()
# model = Perceptron()
model = Sequential()
model.add(Dense(16, input_dim=2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_data, y_data, batch_size=1, epochs=100)



#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과, y_predict", y_predict)
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과, y_predict [0 0 0 1] 

results = model.evaluate(x_data, y_data)
print('metrics.score : ', results[1])
# model.score :  1.0

# acc = accuracy_score(y_data, y_predict)
# print('accuracy_score : ', acc)
# accuracy_score :  1.0



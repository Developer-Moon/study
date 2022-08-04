from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron


#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] # OR gate (합연산)
y_data = [0, 1, 1, 1]


#2. 모델
# model = LinearSVC()
model = Perceptron()


#3. 훈련
model.fit(x_data, y_data)


#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과, y_predict", y_predict)
# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과, y_predict [0 0 0 1] 

results = model.score(x_data, y_data)
print('model.score : ', results)
# model.score :  1.0

acc = accuracy_score(y_data, y_predict)
print('accuracy_score : ', acc)
# accuracy_score :  1.0



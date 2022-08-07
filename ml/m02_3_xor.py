from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC


#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] # XOR gate 같으면 0 틀리면 1
y_data = [0, 1, 1, 0]


#2. 모델
# model = LinearSVC()
model = Perceptron()


#3. 훈련
model.fit(x_data, y_data)


#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과, y_predict", y_predict)

results = model.score(x_data, y_data)
print('model.score :', results)

acc = accuracy_score(y_data, y_predict)
print('accuracy :', acc)

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과, y_predict [1 1 1 1] ----- model = LinearSVC()
# model.score : 0.5
# accuracy : 0.5

# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과, y_predict [0 0 0 0] ----- model = Perceptron()
# model.score : 0.5
# accuracy : 0.5

# 인공지능의 겨울이 11년동안 계속된 이유
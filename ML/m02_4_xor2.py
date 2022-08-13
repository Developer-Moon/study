from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0, 1, 1, 0] # xor: 같으면 0 다르면 1


#2. 모델
model = SVC()


#3. 훈련
model.fit(x_data, y_data)


#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과", y_predict)

result = model.score(x_data, y_data)
print("model.score :", result)

acc = accuracy_score(y_data, y_predict)
print("acc score :", acc)
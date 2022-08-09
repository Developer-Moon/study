from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC # Support Vector Classifier - 레거시안 사이킷런 모델, 원핫 X, 컴파일 X, argmax X

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data 
y = datasets.target 

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
model = LinearSVC()


# 3. 컴파일, 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score = model.score(x_test, y_test)
ypred = model.predict(x_test)

print('acc score: ', score)
print('y_pred: ', ypred)

# acc score:  0.7280701754385965
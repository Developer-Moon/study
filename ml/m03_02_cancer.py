from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 논리회귀(분류)
from sklearn.neighbors import KNeighborsClassifier              # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier             # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


# 2. 모델구성
model = RandomForestClassifier()


# 3. 컴파일, 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score = model.score(x_test, y_test)
print('acc :', score)

# LinearSVC - acc : 0.8508771929824561
# SVC - acc : 0.8947368421052632
# Perceptron - acc : 0.8947368421052632
# LogisticRegression - acc : 0.956140350877193 --------- BEST
# KNeighborsClassifier - acc : 0.9210526315789473
# DecisionTreeClassifier - acc : 0.9122807017543859
# RandomForestClassifier - acc : 0.956140350877193 ----- BEST


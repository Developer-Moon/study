from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 논리회귀(분류)
from sklearn.neighbors import KNeighborsClassifier              # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier             # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)


#2. 모델구성
models = [LinearSVC, SVC, Perceptron, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

for i in models:
    model = i()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    model_name = str(model) # str(model)을 model 가능
    print(model_name, '- acc :', score)

# LinearSVC() - acc : 0.9444444444444444
# SVC() - acc : 0.75
# Perceptron() - acc : 0.5833333333333334
# LogisticRegression() - acc : 1.0
# KNeighborsClassifier() - acc : 0.8055555555555556
# DecisionTreeClassifier() - acc : 0.9722222222222222
# RandomForestClassifier() - acc : 1.0
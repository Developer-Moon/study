from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 논리회귀(분류)
from sklearn.neighbors import KNeighborsClassifier              # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier             # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)


# 2. 모델구성
models = [LinearSVC, SVC, Perceptron, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]

for x in models:
    model = x()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    model_name = str(model)
    print(model_name, '- acc :', score)

# LinearSVC() - acc : 0.9444444444444444
# SVC() - acc : 0.6666666666666666      
# Perceptron() - acc : 0.5555555555555556
# LogisticRegression() - acc : 1.0
# KNeighborsClassifier() - acc : 0.7222222222222222
# DecisionTreeClassifier() - acc : 0.8888888888888888
# RandomForestClassifier() - acc : 1.0
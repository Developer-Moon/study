from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 논리회귀(분류)
from sklearn.neighbors import KNeighborsClassifier              # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier             # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_digits()
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
     
# LinearSVC() acc:  0.95
# SVC() acc:  0.9888888888888889 ------------------ BEST  
# Perceptron() acc:  0.9388888888888889
# LogisticRegression() acc:  0.9611111111111111
# KNeighborsClassifier() acc:  0.9944444444444445
# DecisionTreeClassifier() acc:  0.85
# RandomForestClassifier() acc:  0.9777777777777777
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 논리회귀(분류)
from sklearn.neighbors import KNeighborsClassifier              # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier             # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
        
                      
#2. 모델구성
models = [LinearSVC, SVC, Perceptron, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier]
for x in models:
    model = x()
    model_name = str(model)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(model_name,' - acc :', score)

# LinearSVC()  - acc : 1.0
# SVC()  - acc : 1.0
# Perceptron()  - acc : 0.9666666666666667
# LogisticRegression()  - acc : 1.0
# KNeighborsClassifier()  - acc : 1.0
# DecisionTreeClassifier()  - acc : 1.0
# RandomForestClassifier()  - acc : 1.0
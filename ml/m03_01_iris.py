from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import sklearn as sk
print(sk.__version__) # 0.24.2

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
                      
#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄

model = RandomForestClassifier()


#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('acc 결과: ', result)
y_predict = model.predict(x_test)

# Perceptron - acc 결과:  0.9666666666666667
# LogisticRegression - acc 결과:  1.0
# KNeighborsClassifier - acc 결과:  1.0
# DecisionTreeClassifier - acc 결과:  1.0
# RandomForestClassifier - acc 결과:  1.0
from sklearn.datasets import load_diabetes
from sympy import mobius
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)


# 2. 모델구성
"""
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # !논리회귀(분류임)!
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 봄
"""

from sklearn.svm import LinearSVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = LogisticRegression()


# 3. 컴파일, 훈련



# 4. 평가, 예측
model = LinearSVR()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
ypred = model.predict(x_test)
print('acc score: ', score) # acc score:  -0.20214645760776406
# print('y_pred: ', ypred)

model = LinearRegression()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
ypred = model.predict(x_test)
print('acc score: ', score) # acc score:  0.6557534150889773
# print('y_pred: ', ypred)

model = KNeighborsRegressor()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
ypred = model.predict(x_test)
print('acc score: ', score) # acc score:  0.5704639112420011

model = DecisionTreeRegressor()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
ypred = model.predict(x_test)
print('acc score: ', score) # acc score:  -0.10247725503410288

model = RandomForestRegressor()
model.fit(x_train, y_train )
score = model.score(x_test, y_test)
ypred = model.predict(x_test)
print('acc score: ', score) # acc score:  0.6091284088892704








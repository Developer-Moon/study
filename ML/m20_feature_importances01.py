from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np
#----------------------------------------------------------------------------------------------------------------#
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier # cmd창에서 tf282gpu에서 pip install xgboost
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)


#2. 모델구성
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)

print("________________________")
print(model,':',model.feature_importances_) # 
# 4번째의 feature(0.90731323)가 가장 중요하다 
# 첫번째 feature(0.01253395)는 중요하지 않으니 빼라?

# DecisionTreeClassifier() : [0.01253395 0.01253395 0.06761888 0.90731323]--------------------[0.01253395 0.01253395 0.5618817  0.4130504 ]
# RandomForestClassifier() : [0.08730281 0.03030594 0.4846094  0.39778185]
# GradientBoostingClassifier() : [0.00079228 0.02354189 0.6415879  0.33407793]
# XGBClassifier() : [0.0089478  0.01652037 0.75273126 0.22180054] 가장 정확하다고 한다
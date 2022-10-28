from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris
import numpy as np
#----------------------------------------------------------------------------------------------------------------#
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape)
x = np.delete(x, 0, axis=1)
print(x.shape)

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
print(model,':',model.feature_importances_)

# DecisionTreeClassifier() : [0.02342261 0.         0.07597484 0.90060255]
# acc : 0.9666666666666667
# delete column acc : 0.9666666666666667-----------------------비슷

# RandomForestClassifier() : [0.08946078 0.02533715 0.46308141 0.42212066]
# acc : 0.9666666666666667
# delete column acc : 0.9333333333333333-----------------------감소

# GradientBoostingClassifier() : [0.00081611 0.02429865 0.57987883 0.39500641]
# acc : 0.9666666666666667
# delete column acc : 0.9666666666666667-----------------------비슷

# XGBClassifier() : [0.0089478  0.01652037 0.75273126 0.22180054]
# acc : 0.9666666666666667
# delete column acc : 0.9666666666666667-----------------------비슷
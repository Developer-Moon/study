from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
#----------------------------------------------------------------------------------------------------------------#
from xgboost.plotting import plot_importance

#1. 데이터
datasets = load_diabetes()
print(datasets['feature_names']) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
x = datasets.data   # 넘파이 배열로 받은 것
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)


#2. 모델구성
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()


#3. 훈련
model.fit(x_train, y_train)
#

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('accuracy_score :', r2)

print("________________________")
print(model,':',model.feature_importances_)

# def plot_feature_importances(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center') 
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)
#     plt.title(model)
    
# plot_feature_importances(model)    
# plt.show()

plot_importance(model)
plt.show()
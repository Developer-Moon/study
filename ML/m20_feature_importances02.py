from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
#----------------------------------------------------------------------------------------------------------------#
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor # cmd창에서 tf282gpu에서 pip install xgboost
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)


#2. 모델구성
model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
# model = XGBRegressor()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('accuracy_score :', r2)

print("________________________")
print(model,':',model.feature_importances_)

# DecisionTreeRegressor() : [0.10212395 0.02483863 0.23186745 0.05197325 0.04855492 0.04306545 0.03534532 0.02493843 0.36587595 0.07141665]
# - model.score : 0.16695989185234772

# RandomForestRegressor() : [0.05631661 0.01199745 0.29900275 0.10354603 0.04164461 0.05292383 0.05350437 0.02695972 0.27077244 0.08333218]
# - model.score : 0.532813772337906   

# GradientBoostingRegressor() : [0.05017833 0.0107929  0.30430634 0.11139135 0.02793022 0.05266135 0.03933056 0.02146657 0.3377114  0.04423098]
# - model.score : 0.559421020838526

# XGBRegressor() : [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819 0.06012432 0.09595273 0.30483875 0.06629313]
# - model.score : 0.4590400803596264
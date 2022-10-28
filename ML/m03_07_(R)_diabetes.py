from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron, LinearRegression 
from sklearn.neighbors import KNeighborsRegressor            
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_diabetes()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)


#2. 모델구성
models = [LinearSVR, SVR, Perceptron, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for i in models:
    model = i()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    model_name = str(model) # str(model)을 model 가능
    print(model_name, '- r2 :', score)

# LinearSVR() - r2 : -0.28343066601535716
# SVR() - r2 : 0.21089225663642996
# Perceptron() - r2 : 0.0
# LinearRegression() - r2 : 0.5851141269959736
# KNeighborsRegressor() - r2 : 0.48322394424037063
# DecisionTreeRegressor() - r2 : -0.2576688721970706
# RandomForestRegressor() - r2 : 0.535805494646879
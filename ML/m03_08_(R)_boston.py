from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron, LinearRegression 
from sklearn.neighbors import KNeighborsRegressor            
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=9)


#2. 모델구성
models = [LinearSVR, SVR, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor] # Perceptron 오류

for i in models:
    model = i()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    model_name = str(model) # str(model)을 model 가능
    print(model_name, '- r2 :', score)

# LinearSVR() - r2 : 0.5604803474127777
# SVR() - r2 : 0.23368969054695032
# LinearRegression() - r2 : 0.7660111574904016
# KNeighborsRegressor() - r2 : 0.6191790652200567
# DecisionTreeRegressor() - r2 : 0.8308770343735064
# RandomForestRegressor() - r2 : 0.8934247115751678
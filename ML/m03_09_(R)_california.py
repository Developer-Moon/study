from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron, LinearRegression 
from sklearn.neighbors import KNeighborsRegressor            
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
datasets = fetch_california_housing()
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
    print(model_name, '- acc :', score)

# LinearSVR() - acc : -7.2604522701281855
# SVR() - acc : -0.02988719784087035
# LinearRegression() - acc : 0.5743251711424016
# KNeighborsRegressor() - acc : 0.14243805092295736
# DecisionTreeRegressor() - acc : 0.6392378112735544
# RandomForestRegressor() - acc : 0.8142219912336345
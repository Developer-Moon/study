from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor            
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor           
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=86)


# 2. 모델구성
model_list = [LinearSVR, SVR, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for x in model_list:
     model = x()
     model.fit(x_train, y_train)
     score = model.score(x_test, y_test)
     model_name = str(model)
     print(model_name, 'acc: ', score) 
     
# LinearSVR() acc:  -8.773570609283057
# SVR() acc:  -0.024792776551759488
# LinearRegression() acc:  0.6051462771302347
# KNeighborsRegressor() acc:  0.1557659976728809
# DecisionTreeRegressor() acc:  0.618452330358615
# RandomForestRegressor() acc:  0.8125532368374353
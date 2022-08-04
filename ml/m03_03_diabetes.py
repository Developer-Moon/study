from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor            
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor           
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)


# 2. 모델구성
model = RandomForestRegressor()


# 3. 컴파일, 훈련
model.fit(x_train, y_train)


# 4. 평가, 예측
score = model.score(x_test, y_test)
print('r2 :', score)

# LinearSVR - r2 : -0.2112008649112267
# SVR - r2 : 0.2668794571758186
# LinearRegression - r2 : 0.6557534150889773 ---------- BEST  
# KNeighborsRegressor - r2 : 0.5704639112420011
# DecisionTreeRegressor - r2 : -0.03424678846812457
# RandomForestRegressor - r2 : 0.604705652707703

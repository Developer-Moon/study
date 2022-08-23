

# 회귀 모델 ------------------------------------------------------------------------------------------------------
from sklearn.svm import LinearSVR, SVR # 원핫 X, 컴파일 X, argmax X

from sklearn.linear_model import Perceptron, LinearRegression 
from sklearn.neighbors import KNeighborsRegressor            
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor             # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다




# 분류 모델 ------------------------------------------------------------------------------------------------------
from sklearn.svm import LinearSVC, SVC # 원핫 X, 컴파일 X, argmax X

from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression : 논리회귀(이진분류)
from sklearn.neighbors import KNeighborsClassifier              # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier             # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다
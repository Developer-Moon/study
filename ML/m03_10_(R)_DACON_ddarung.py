from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import Perceptron, LinearRegression 
from sklearn.neighbors import KNeighborsRegressor            
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set = train_set.fillna(0)

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  


#2. 모델구성
models = [LinearSVR, SVR, Perceptron, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor] # Perceptron 오류

for i in models:
    model = i()
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    model_name = str(model) # str(model)을 model 가능
    print(model_name, '- acc :', score)

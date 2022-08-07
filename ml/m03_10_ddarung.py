from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
import pandas as pd
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor            
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor           
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터 
path = './_data/ddarung/'                                       
train_set = pd.read_csv(path + 'train.csv', index_col=0)                                                               
test_set = pd.read_csv(path + 'test.csv', index_col=0) 

train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   
                                                                                           
train_set = train_set.dropna() 

x = train_set.drop(['count'], axis=1)        
y = train_set['count']                  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=16)

scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model_list = [LinearSVR, SVR, LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for x in model_list:
     model = x()
     model.fit(x_train, y_train)
     score = model.score(x_test, y_test)
     model_name = str(model)
     print(model_name, 'acc: ', score) 

# LinearSVR() acc:  0.5365779516822733
# SVR() acc:  0.42663163184723396
# LinearRegression() acc:  0.6366061065665973
# KNeighborsRegressor() acc:  0.6696416633552421
# DecisionTreeRegressor() acc:  0.5330388566008688
# RandomForestRegressor() acc:  0.7561376243432043
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 
from sklearn.svm import LinearSVR                         
import pandas as pd


#1. 데이터
path = './_data/kaggle_bike/'        
train_set = pd.read_csv(path + 'train.csv', index_col=0)   
test_set = pd.read_csv(path + 'test.csv', index_col=0)  

x = train_set.drop(['casual', 'registered', 'count'], axis=1)  
y = train_set['count']   
  
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)  
    
scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = LinearSVR()


#3. 컴파일, 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)             
print('r2 :', results)

# r2 : 0.23773122662776558
# ML - r2 : 0.19437234933998793
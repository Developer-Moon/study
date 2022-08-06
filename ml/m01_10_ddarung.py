from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.svm import LinearSVR
import pandas as pd


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
model = LinearSVR()


#3. 컴파일, 훈련
model.fit(x_train, y_train) 


#4. 평가, 예측
results = model.score(x_test, y_test)
print('r2 :', results)            

# r2 : 0.6209826631438036
# ML - r2 : 0.5356306305873728
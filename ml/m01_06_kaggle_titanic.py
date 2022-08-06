from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC


# 1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'], axis=1)
y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

scaler = RobustScaler()
scaler.fit(x_train)                     
x_train = scaler.transform(x_train)    
x_test = scaler.transform(x_test)      
test_set = scaler.transform(test_set)  



# 2. 모델 구성
model = LinearSVC()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
print('acc :', results)

# acc : 0.8666666666666667
# 머신러닝 사용 - acc : 0.7888888888888889
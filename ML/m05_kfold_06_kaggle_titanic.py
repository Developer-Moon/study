from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
import pandas as pd
import numpy as np

# 1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

print(train_set.describe())
print(train_set.info())
print(train_set.isnull())
print(train_set.isnull().sum())
print(train_set.shape) # (10886, 12)
print(train_set.columns.values) # ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
print(train_set['Embarked'].mode())
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
print(test_set.isnull().sum())
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
print(test_set['Embarked'].mode())
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) 
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5              
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66 )

#2. 모델구성
model = SVC()


#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold) # cv=5 라면 kfold를 5로 쓴다
print('ACC :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)

acc = accuracy_score(y_test, y_predict)
print('cross_val_predict ACC : ', acc)

# ACC : [0.86013986 0.8041958  0.77464789 0.83098592 0.80985915] 
# cross_val_score : 0.816
# [1 0 0 1 1 0 0 0 0 1 0 1 0 0 1 0 0 1 1 0 0 0 1 0 0 1 0 0 0 1 0 0 1 1 0 0 1
#  0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 1 0 0 1 1 0 1 1 1 1 0 1 1 1 0 0 0 0 0 0 1
#  0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0 1 1 0 1 0 0 0 1 0 0 0 1 1 0 0 1 1 0 0 0
#  1 0 1 1 1 1 1 0 1 0 0 1 0 1 1 0 0 1 0 0 1 1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 0
#  1 0 0 0 1 1 1 0 0 1 1 0 1 0 1 0 0 0 1 1 0 0 0 0 1 1 0 1 0 1 1]
# cross_val_predict ACC :  0.7094972067039106
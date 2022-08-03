import numpy as np
import pandas as pd
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

from sklearn.svm import LinearSVC


# dataset.describe()
# dataset.info()
# dataset.isnull().sum()
# pandas의 y라벨의 종류가 무엇인지 확인하는 함수 쓸 것
# numpy에서는 np.unique(y, return_counts=True) => pandas에서 동일한 함수 확인     

# 1. 데이터

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)

print(train_set)
print(train_set.columns)                         
print('train_set의 info :: ', train_set.info())
print(train_set.describe())
print(train_set.isnull().sum())     # Age 177, Cabin 687, Embarked 2
print("========================================================")

test_set = pd.read_csv(path + 'test.csv', index_col=0)

# print(test_set) 
# print(test_set.columns) 
# print('test_set의 info :: ', test_set.info())
# # print(test_set.describe())
# print(test_set.feature_name)
# print(test_set.isnull().sum())  # Age 86, Cabin 327  
# print("========================================================")

submission = pd.read_csv(path + 'gender_submission.csv')
print('train.shape, test.shape, submit.shape', 
      train_set.shape, test_set.shape, submission.shape)    # (891, 11) (418, 10) (418, 2)

# 데이터 전처리

# train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train_set[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train_set[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train_set[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Ticket, Cabin[선실], Name 삭제   
train_set = train_set.drop(['Ticket', 'Cabin', 'Name'], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin', 'Name'], axis=1)


# Age NaN값 변환
train_set['Age'] = train_set['Age'].fillna(train_set.Age.dropna().mode()[0])  # NaN값을 최빈값으로 채운다
test_set['Age'] = test_set['Age'].fillna(train_set.Age.dropna().mode()[0])


# Embarked, Sex NaN값 및 Object => int 변환
train_set['Embarked'] = train_set['Embarked'].fillna(train_set.Embarked.dropna().mode()[0])    # Embarked 부분을 지우고 
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)           # 'S':0, 'C':1, 'Q':2 로 매핑한다 int로
test_set['Embarked'] = test_set['Embarked'].fillna(test_set.Embarked.dropna().mode()[0])
test_set['Embarked'] = test_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

train_set['Sex'] = train_set['Sex'].fillna(train_set.Sex.dropna().mode()[0])
train_set['Sex'] = train_set['Sex'].map({'male':0, 'female':1}).astype(int)
test_set['Sex'] = test_set['Sex'].fillna(test_set.Sex.dropna().mode()[0])
test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1}).astype(int)

print(train_set.shape, test_set.shape)  # (891, 8) (418, 7)
print(train_set.head(5))
print(test_set.head(5))
print(train_set.isnull().sum())  
print(test_set.isnull().sum())  

# x, y 데이터
x = train_set.drop(['Survived'], axis=1)
print(x)
print(x.columns)    # 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)

# One Hot Encoding
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape)  # (891, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=3
)


# scaler = MinMaxScaler() 
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)                      # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)        # train이 변한 범위에 맞춰서 변환됨
test_set = scaler.transform(test_set)  
# y_summit = model.predict(test_set) test셋은 스케일링이 상태가 아니니 summit전에 스케일링을 해서  y_summit = model.predict(test_set) 에 넣어줘야 한다 
# summit하기 전에만 해주면 상관이 없다

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위






# 2. 모델 구성
model = LinearSVC()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print('acc :', results)



#5. 데이터 summit
# y_summit = model.predict(test_set)
# y_summit = y_summit.flatten()                 
# y_summit = np.where(y_summit > 0.5, 1 , 0)   
# # print(y_summit)
# # print(y_summit.shape)

# submission = pd.read_csv('./_data/kaggle_titanic/submission.csv')
# submission['Survived'] = y_summit
# print(submission)
# submission.to_csv('./_data/kaggle_titanic/submission1.csv', index=False)

#==================================================================================#
# random_state=3
# loss :  0.39357537031173706
# acc 스코어 :  acc 스코어 :  0.8666666666666667
#==================================================================================#
'''
[scaler = MinMaxScaler]
loss      : 0.5917669534683228
acc 스코어 :  0.7666666666666667

[scaler = StandardScaler]
loss      :  0.7088110446929932
acc 스코어 :  0.7666666666666667

[scaler = MaxAbsScaler]

loss      : 0.548823893070221
acc 스코어 :  0.7555555555555555            

[scaler = RobustScaler]                             함수모델 사용
loss      : 0.6145671606063843
acc 스코어 :  0.7777777777777778                    acc 스코어 :  0.8
'''

# 머신러닝 사용 acc : 0.7888888888888889




































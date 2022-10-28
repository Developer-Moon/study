from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.covariance import EllipticEnvelope
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # simple linear
import seaborn as sns


# 1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')


print(train_set.isnull().sum())
print(train_set.corr())


#              PassengerId  Survived    Pclass       Age     SibSp     Parch      Fare
# PassengerId     1.000000 -0.005007 -0.035144  0.036847 -0.057527 -0.001652  0.012658
# Survived       -0.005007  1.000000 -0.338481 -0.077221 -0.035322  0.081629  0.257307
# Pclass         -0.035144 -0.338481  1.000000 -0.369226  0.083081  0.018443 -0.549500
# Age             0.036847 -0.077221 -0.369226  1.000000 -0.308247 -0.189119  0.096067
# SibSp          -0.057527 -0.035322  0.083081 -0.308247  1.000000  0.414838  0.159651
# Parch          -0.001652  0.081629  0.018443 -0.189119  0.414838  1.000000  0.216225
# Fare            0.012658  0.257307 -0.549500  0.096067  0.159651  0.216225  1.000000






train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'], axis=1)
y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5              
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66 )
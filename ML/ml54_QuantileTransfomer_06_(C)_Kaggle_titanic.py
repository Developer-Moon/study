from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#----------------------------------------------------------------------------------------#
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다
#----------------------------------------------------------------------------------------#


#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) 
y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9, stratify=y)

scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(),
           PowerTransformer(method='yeo-johnson'), PowerTransformer(method='box-cox')]
models = [LGBMClassifier(), CatBoostClassifier(verbose=0), XGBClassifier(), RandomForestClassifier()]

for i in scalers:
    scaler = i
    if str(scaler) == str(PowerTransformer(method='box-cox')): # 이 부분이 좀 이해가 안 간다...
        try:
            x_train = scaler.fit_transform(x_train) 
        except:
            print('진짜 도덕책이네...')
            break

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for model in models:       
        model = model
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        result = accuracy_score(y_test, y_predict)   
        
        if str(model).startswith('<cat'):
            print('CatBoostClassifier -', str(scaler).replace('()',' :'), round(result, 4))
        elif str(model).startswith('XGB'):
            print('XGBClassifier -', str(scaler).replace('()',' :'), round(result, 4))
        else:
            print(str(model).replace('()',' -'), str(scaler).replace('()',' :'), round(result, 4))    
        
# LGBMClassifier - StandardScaler : 0.8101
# CatBoostClassifier - StandardScaler : 0.8268
# XGBClassifier - StandardScaler : 0.8045
# RandomForestClassifier - StandardScaler : 0.7989
# LGBMClassifier - MinMaxScaler : 0.8101
# CatBoostClassifier - MinMaxScaler : 0.8268
# XGBClassifier - MinMaxScaler : 0.8045
# RandomForestClassifier - MinMaxScaler : 0.7989
# LGBMClassifier - MaxAbsScaler : 0.8101
# CatBoostClassifier - MaxAbsScaler : 0.8268
# XGBClassifier - MaxAbsScaler : 0.8045
# RandomForestClassifier - MaxAbsScaler : 0.8045
# LGBMClassifier - RobustScaler : 0.8212
# CatBoostClassifier - RobustScaler : 0.8268
# XGBClassifier - RobustScaler : 0.8045
# RandomForestClassifier - RobustScaler : 0.7989
# LGBMClassifier - QuantileTransformer : 0.8101
# CatBoostClassifier - QuantileTransformer : 0.8268
# XGBClassifier - QuantileTransformer : 0.8045
# RandomForestClassifier - QuantileTransformer : 0.8268      
# LGBMClassifier - PowerTransformer : 0.8101
# CatBoostClassifier - PowerTransformer : 0.8268
# XGBClassifier - PowerTransformer : 0.8045
# RandomForestClassifier - PowerTransformer : 0.7989
# 진짜 도덕책이네...
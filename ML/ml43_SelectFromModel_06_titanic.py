from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import fetch_covtype
from xgboost import XGBClassifier,XGBRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

#1.데이터 
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


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)

parameters = {'n_estimators' : [100],
              'learning_rate': [0.1],
              'max_depth': [3],
              'gamma': [1],
              'min_child_weight': [1],
              'subsample': [1],
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              'colsample_bynode': [1] ,
              'reg_alpha': [0],
              'reg_lambda':[1]
              }

#2.모델 
model = XGBClassifier(random_state=123,
                      n_estimators=1000,
                      learning_rate=0.1,
                      max_depth=3,
                      gamma=1,
                    )

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train, y_train, # early_stopping_rounds=100,
          # eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric='error', 
          # 회귀 : rmse, mae, resle
          # 이진 : error, auc...mlogloss...
          # 다중이 : merror, mlogloss...
          ) 

results = model.score(x_test, y_test)
print('최종점수 :', results) 

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('진짜 최종점수 test 점수 :', acc)

print(model.feature_importances_)
# [0.         0.         0.7567438  0.24325623]

thresholds = model.feature_importances_
print('___________________________')
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # prefit 크거나 같은 컬럼을 빼준다
  
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)  
    
    selection_model = XGBClassifier(n_jobs=-1,   # 훈련이 11번?
                                   random_state=123,
                                   n_estimators=100,
                                   learning_rate=0.1,
                                   max_depth=3,
                                   gamma=1)
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    
    print("Thresh = %.3f, n=%d, accuracy: %.2f%% "
          %(thresh, select_x_train.shape[1], score*100))
    
# (712, 5) (179, 5)
# Thresh = 0.046, n=5, accuracy: 77.65% 
    
    
# (712, 2) (179, 2)
# Thresh = 0.158, n=2, accuracy: 72.63% 
# (712, 1) (179, 1)
# Thresh = 0.609, n=1, accuracy: 74.30% 
# (712, 5) (179, 5)
# Thresh = 0.046, n=5, accuracy: 77.65% 
# (712, 3) (179, 3)
# Thresh = 0.067, n=3, accuracy: 74.30% 
# (712, 7) (179, 7)
# Thresh = 0.030, n=7, accuracy: 77.65% 
# (712, 4) (179, 4)
# Thresh = 0.049, n=4, accuracy: 74.86% 
# (712, 6) (179, 6)
# Thresh = 0.041, n=6, accuracy: 77.09% 

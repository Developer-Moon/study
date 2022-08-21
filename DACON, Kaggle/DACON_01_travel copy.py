import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import time
import seaborn as sns
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from imblearn.over_sampling import SMOTE                                     
from bayes_opt import BayesianOptimization

import warnings
warnings.filterwarnings(action='ignore')

# 1. 데이터
path  = './_data/dacon_travel/'      
train = pd.read_csv(path + 'train.csv', index_col=0)
test  = pd.read_csv(path + 'test.csv', index_col=0)


# print(train.info())
'''
0   Age                       1861 non-null   float64
1   TypeofContact             1945 non-null   object
2   CityTier                  1955 non-null   int64
3   DurationOfPitch           1853 non-null   float64
4   Occupation                1955 non-null   object
5   Gender                    1955 non-null   object
6   NumberOfPersonVisiting    1955 non-null   int64
7   NumberOfFollowups         1942 non-null   float64
8   ProductPitched            1955 non-null   object
9   PreferredPropertyStar     1945 non-null   float64
10  MaritalStatus             1955 non-null   object
11  NumberOfTrips             1898 non-null   float64
12  Passport                  1955 non-null   int64
13  PitchSatisfactionScore    1955 non-null   int64
14  OwnCar                    1955 non-null   int64
15  NumberOfChildrenVisiting  1928 non-null   float64
16  Designation               1955 non-null   object
17  MonthlyIncome             1855 non-null   float64
18  ProdTaken                 1955 non-null   int64
'''

le = LabelEncoder() 
train_cols = np.array(train.columns)

for i in train_cols:
    if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])

'''
sns.set(font_scale= 0.8 )
sns.heatmap(data=train.corr(), square= True, annot=True, cbar=True) # square: 정사각형, annot: 안에 수치들 ,cbar: 옆에 bar
plt.show() 

Age = -0.14
ProductPitched = -0.15
MonthlyIncome  = -0.14
'''


# print(test.isnull().sum())
train = train.dropna()

test['DurationOfPitch'].fillna(test['DurationOfPitch'].mean(), inplace=True) 
test['NumberOfFollowups'].fillna(test['NumberOfFollowups'].mean(), inplace=True) 
test['PreferredPropertyStar'].fillna(test['PreferredPropertyStar'].median(), inplace=True) 
test['NumberOfTrips'].fillna(test['NumberOfTrips'].median(), inplace=True) 
test['NumberOfChildrenVisiting'].fillna(test['NumberOfChildrenVisiting'].median(), inplace=True) 

x = train.drop(['ProdTaken', 'Age', 'MonthlyIncome'], axis=1)
y = train['ProdTaken']
test = test.drop(['Age', 'MonthlyIncome'], axis=1)


# print(x.isnull().sum())

# TypeofContact               0
# CityTier                    0
# DurationOfPitch             0
# Occupation                  0
# Gender                      0
# NumberOfPersonVisiting      0
# NumberOfFollowups           0
# ProductPitched              0
# PreferredPropertyStar       0
# MaritalStatus               0
# NumberOfTrips               0
# Passport                    0
# PitchSatisfactionScore      0
# OwnCar                      0
# NumberOfChildrenVisiting    0
# Designation                 0


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=40, stratify=y)

                                                                       
#2. 모델 
bayesian_params = {
    'colsample_bytree' : (0.5, 0.7),
    'max_depth' : (10,18),
    'min_child_weight' : (30, 35),
    'reg_alpha' : (43, 47),
    'reg_lambda' : (0.001, 0.01),
    'subsample' : (0.4, 0.7)
}

{'target': 0.793939393939394, 'params': {'colsample_bytree': 0.5383038900757784, 'max_depth': 14.976870168318655, 'min_child_weight': 32.18863869503557, 'reg_alpha': 46.14143433485508,
                                         'reg_lambda': 0.008019782273069233, 'subsample': 0.4817777815847925}}



def xgb_function(max_depth, min_child_weight,subsample, colsample_bytree, reg_lambda,reg_alpha):
    params ={
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                    # 정수만
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),                  # 0~1 사이값만
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'reg_lambda' : max(reg_lambda,0),                       # 양수만
        'reg_alpha' : max(reg_alpha,0),
    }
            
            
        # 'num_leaves':int(round(num_leaves)),
        # 'min_child_samples':int(round(min_child_samples)),
        # 'min_child_weight':int(round(min_child_weight)), # 무조건 정수
        # 'subsample':max(min(subsample, 1), 0),  # 0~1 사이의 값이 들어와야 한다 1이상이면 1
        # 'colsample_bytree':max(min(colsample_bytree, 1), 0),
        # 'max_bin':max(int(round(max_bin)), 10), # 10이상의 정수
        # 'reg_lambda':max(reg_lambda, 0),        # 무조건 양수만
        # 'reg_alpha':max(reg_alpha, 0)           # 무조건 양수만
    
    # *여러개의인자를받겠다     
    # **키워드받겠다(딕셔너리형태)
    
    model = XGBClassifier(**params) # 모델을 함수안에 써야한다
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    
    return result

lgb_bo = BayesianOptimization(f=xgb_function,
                              pbounds=bayesian_params,
                              random_state=1234)

lgb_bo.maximize(init_points=5, n_iter=110)

print(lgb_bo.max)     


                                                                                                   
'''
#2. 모델구성
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                      )

# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, n_iter=15)



#3. 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)

# 5. 제출 준비
# y_submit = model.predict(test)

# submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
# submission['ProdTaken'] = y_submit
# submission.to_csv(path + 'sample_submission2.csv', index = True)




# model.score : 0.8900255754475703
# accuracy_score : 0.8900255754475703

'''                                                                                                       

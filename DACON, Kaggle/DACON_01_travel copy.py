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



from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] 

test_set.drop(drop_cols, axis = 1, inplace =True)
train_set.drop(drop_cols, axis = 1, inplace =True)

cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

train_set = train_set.fillna(train_set.mean()) 
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=68)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
             
             
#2. 모델구성
# all_Algorithms = all_estimators(type_filter='classifier')  # 분류모델
all_Algorithms = all_estimators(type_filter='regressor') # 회귀모델
print('allAlgorithms: ', all_Algorithms)
print('모델의 개수: ', len(all_Algorithms)) # 41

for (name, algorithm) in all_Algorithms:
    try:
        model = algorithm()
        ypred = cross_val_predict(model, x_test, y_test, cv=kfold)
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률: ', round(np.mean(score),4))
        
    except:
        # continue # 또는 pass
        print(name, '은 안나온 놈')

# ARDRegression 의 정답률:  0.763
# AdaBoostRegressor 의 정답률:  0.799
# BaggingRegressor 의 정답률:  0.826
# BayesianRidge 의 정답률:  0.7726
# CCA 의 정답률:  -0.5844
# DecisionTreeRegressor 의 정답률:  0.7053
# DummyRegressor 의 정답률:  -0.0041
# ElasticNet 의 정답률:  0.5325
# ElasticNetCV 의 정답률:  0.0441
# ExtraTreeRegressor 의 정답률:  0.6625
# ExtraTreesRegressor 의 정답률:  0.843
# GammaRegressor 의 정답률:  0.4475
# GaussianProcessRegressor 의 정답률:  0.5815
# GradientBoostingRegressor 의 정답률:  0.8733
# HistGradientBoostingRegressor 의 정답률:  0.8625
# HuberRegressor 의 정답률:  0.8102
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.7015
# KernelRidge 의 정답률:  0.7757
# Lars 의 정답률:  0.443
# LarsCV 의 정답률:  0.749
# Lasso 의 정답률:  0.7483
# LassoCV 의 정답률:  0.7836
# LassoLars 의 정답률:  0.749
# LassoLarsCV 의 정답률:  0.7551
# LassoLarsIC 의 정답률:  0.7689
# LinearRegression 의 정답률:  0.7479
# LinearSVR 의 정답률:  -4.0534
# MLPRegressor 의 정답률:  -4.8002
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  -0.0191
# OrthogonalMatchingPursuit 의 정답률:  0.7446
# OrthogonalMatchingPursuitCV 의 정답률:  0.7302
# PLSCanonical 의 정답률:  -4.9004
# PLSRegression 의 정답률:  0.7788
# PassiveAggressiveRegressor 의 정답률:  0.7746
# PoissonRegressor 의 정답률:  0.7686
# RANSACRegressor 의 정답률:  -2.8664586407417033e+19
# RadiusNeighborsRegressor 의 정답률:  -9.651135186141302e+27
# RandomForestRegressor 의 정답률:  0.8487
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.7749
# RidgeCV 의 정답률:  0.7724
# SGDRegressor 의 정답률:  0.7996
# SVR 의 정답률:  -0.0626
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.7631
# TransformedTargetRegressor 의 정답률:  0.7479
# TweedieRegressor 의 정답률:  0.391
# VotingRegressor 은 안나온 놈
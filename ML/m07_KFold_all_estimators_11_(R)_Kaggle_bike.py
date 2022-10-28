from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
import numpy as np
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True)
test_set.drop('datetime',axis=1,inplace=True)
train_set.drop('casual',axis=1,inplace=True) 
train_set.drop('registered',axis=1,inplace=True) 

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

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

# ARDRegression 의 정답률:  0.3883
# AdaBoostRegressor 의 정답률:  0.6805
# BaggingRegressor 의 정답률:  0.9383
# BayesianRidge 의 정답률:  0.3884
# CCA 의 정답률:  0.0583
# DecisionTreeRegressor 의 정답률:  0.8877
# DummyRegressor 의 정답률:  -0.0006
# ElasticNet 의 정답률:  0.1392
# ElasticNetCV 의 정답률:  0.3744
# ExtraTreeRegressor 의 정답률:  0.8749
# ExtraTreesRegressor 의 정답률:  0.9467
# GammaRegressor 의 정답률:  0.0577
# GaussianProcessRegressor 의 정답률:  -30.8363
# GradientBoostingRegressor 의 정답률:  0.8711
# HistGradientBoostingRegressor 의 정답률:  0.9528
# HuberRegressor 의 정답률:  0.3565
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.6639
# KernelRidge 의 정답률:  0.3883
# Lars 의 정답률:  0.3884
# LarsCV 의 정답률:  0.3885
# Lasso 의 정답률:  0.3856
# LassoCV 의 정답률:  0.3887
# LassoLars 의 정답률:  -0.0006
# LassoLarsCV 의 정답률:  0.3885
# LassoLarsIC 의 정답률:  0.388
# LinearRegression 의 정답률:  0.3884
# LinearSVR 의 정답률:  0.3151
# MLPRegressor 의 정답률:  0.4687
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.331
# OrthogonalMatchingPursuit 의 정답률:  0.162
# OrthogonalMatchingPursuitCV 의 정답률:  0.3868
# PLSCanonical 의 정답률:  -0.3359
# PLSRegression 의 정답률:  0.3833
# PassiveAggressiveRegressor 의 정답률:  0.3105
# PoissonRegressor 의 정답률:  0.4179
# RANSACRegressor 의 정답률:  0.1213
# RadiusNeighborsRegressor 의 정답률:  0.223
# RandomForestRegressor 의 정답률:  0.9457
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.3884
# RidgeCV 의 정답률:  0.3882
# SGDRegressor 의 정답률:  0.3875
# SVR 의 정답률:  0.3222
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.3846
# TransformedTargetRegressor 의 정답률:  0.3884
# TweedieRegressor 의 정답률:  0.086
# VotingRegressor 은 안나온 놈
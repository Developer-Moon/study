from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import fetch_california_housing
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
         
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
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

# ARDRegression 의 정답률:  0.6101
# AdaBoostRegressor 의 정답률:  0.4375
# BaggingRegressor 의 정답률:  0.7882
# BayesianRidge 의 정답률:  0.6099
# CCA 의 정답률:  0.5655
# DecisionTreeRegressor 의 정답률:  0.5959
# DummyRegressor 의 정답률:  -0.0005
# ElasticNet 의 정답률:  -0.0005
# ElasticNetCV 의 정답률:  0.6012
# ExtraTreeRegressor 의 정답률:  0.5922
# ExtraTreesRegressor 의 정답률:  0.8093
# GammaRegressor 의 정답률:  0.0189
# GaussianProcessRegressor 의 정답률:  -14011.6662
# GradientBoostingRegressor 의 정답률:  0.7869
# HistGradientBoostingRegressor 의 정답률:  0.8344
# HuberRegressor 의 정답률:  0.575
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.701
# KernelRidge 의 정답률:  0.5341
# Lars 의 정답률:  0.6099
# LarsCV 의 정답률:  0.6093
# Lasso 의 정답률:  -0.0005
# LassoCV 의 정답률:  0.6092
# LassoLars 의 정답률:  -0.0005
# LassoLarsCV 의 정답률:  0.6093
# LassoLarsIC 의 정답률:  0.6099
# LinearRegression 의 정답률:  0.6099
# LinearSVR 의 정답률:  0.586
# MLPRegressor 의 정답률:  0.7253
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.6649
# OrthogonalMatchingPursuit 의 정답률:  0.4751
# OrthogonalMatchingPursuitCV 의 정답률:  0.6013
# PLSCanonical 의 정답률:  0.3688
# PLSRegression 의 정답률:  0.5235
# PassiveAggressiveRegressor 의 정답률:  0.2335
# PoissonRegressor 의 정답률:  0.0409
# RANSACRegressor 의 정답률:  -1.4448
# RadiusNeighborsRegressor 의 정답률:  0.0139
# RandomForestRegressor 의 정답률:  0.8054
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.6032
# RidgeCV 의 정답률:  0.6076
# SGDRegressor 의 정답률:  0.5627
# SVR 의 정답률:  0.6612
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  -3.1499
# TransformedTargetRegressor 의 정답률:  0.6099
# TweedieRegressor 의 정답률:  0.019
# VotingRegressor 은 안나온 놈
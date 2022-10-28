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
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set = train_set.fillna(0)

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

scaler = RobustScaler()
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

# ARDRegression 의 정답률:  0.5915
# AdaBoostRegressor 의 정답률:  0.6062
# BaggingRegressor 의 정답률:  0.7575
# BayesianRidge 의 정답률:  0.5914
# CCA 의 정답률:  0.2324
# DecisionTreeRegressor 의 정답률:  0.5851
# DummyRegressor 의 정답률:  -0.0061
# ElasticNet 의 정답률:  0.4987
# ElasticNetCV 의 정답률:  0.586
# ExtraTreeRegressor 의 정답률:  0.5416
# ExtraTreesRegressor 의 정답률:  0.7976
# GammaRegressor 의 정답률:  0.3186
# GaussianProcessRegressor 의 정답률:  -0.2464
# GradientBoostingRegressor 의 정답률:  0.7691
# HistGradientBoostingRegressor 의 정답률:  0.7788
# HuberRegressor 의 정답률:  0.5805
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.6476
# KernelRidge 의 정답률:  -0.9143
# Lars 의 정답률:  0.5915
# LarsCV 의 정답률:  0.5912
# Lasso 의 정답률:  0.5865
# LassoCV 의 정답률:  0.5914
# LassoLars 의 정답률:  0.3468
# LassoLarsCV 의 정답률:  0.5912
# LassoLarsIC 의 정답률:  0.5921
# LinearRegression 의 정답률:  0.5915
# LinearSVR 의 정답률:  0.5196
# MLPRegressor 의 정답률:  0.5644
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.3852
# OrthogonalMatchingPursuit 의 정답률:  0.3729
# OrthogonalMatchingPursuitCV 의 정답률:  0.5759
# PLSCanonical 의 정답률:  -0.3589
# PLSRegression 의 정답률:  0.5891
# PassiveAggressiveRegressor 의 정답률:  0.546
# PoissonRegressor 의 정답률:  0.6112
# RANSACRegressor 의 정답률:  0.505
# RadiusNeighborsRegressor 의 정답률:  nan
# RandomForestRegressor 의 정답률:  0.7798
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.5915
# RidgeCV 의 정답률:  0.5915
# SGDRegressor 의 정답률:  0.5911
# SVR 의 정답률:  0.3927
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.5749
# TransformedTargetRegressor 의 정답률:  0.5915
# TweedieRegressor 의 정답률:  0.4291
# VotingRegressor 은 안나온 놈
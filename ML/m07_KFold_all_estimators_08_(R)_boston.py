from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_boston
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = load_boston()
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

# ARDRegression 의 정답률:  0.6844
# AdaBoostRegressor 의 정답률:  0.7503
# BaggingRegressor 의 정답률:  0.7447
# BayesianRidge 의 정답률:  0.6918
# CCA 의 정답률:  0.6152
# DecisionTreeRegressor 의 정답률:  0.5406
# DummyRegressor 의 정답률:  -0.0261
# ElasticNet 의 정답률:  0.121
# ElasticNetCV 의 정답률:  0.6918
# ExtraTreeRegressor 의 정답률:  0.5611
# ExtraTreesRegressor 의 정답률:  0.8394
# GammaRegressor 의 정답률:  0.1603
# GaussianProcessRegressor 의 정답률:  -1.467
# GradientBoostingRegressor 의 정답률:  0.8026
# HistGradientBoostingRegressor 의 정답률:  0.8032
# HuberRegressor 의 정답률:  0.6765
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.6561
# KernelRidge 의 정답률:  0.6004
# Lars 의 정답률:  0.664
# LarsCV 의 정답률:  0.6739
# Lasso 의 정답률:  0.1896
# LassoCV 의 정답률:  0.6894
# LassoLars 의 정답률:  -0.0261
# LassoLarsCV 의 정답률:  0.6903
# LassoLarsIC 의 정답률:  0.6681
# LinearRegression 의 정답률:  0.6909
# LinearSVR 의 정답률:  0.5808
# MLPRegressor 의 정답률:  0.1529
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.5288
# OrthogonalMatchingPursuit 의 정답률:  0.5235
# OrthogonalMatchingPursuitCV 의 정답률:  0.6614
# PLSCanonical 의 정답률:  -2.2878
# PLSRegression 의 정답률:  0.673
# PassiveAggressiveRegressor 의 정답률:  0.6078
# PoissonRegressor 의 정답률:  0.6014
# RANSACRegressor 의 정답률:  0.4838
# RadiusNeighborsRegressor 의 정답률:  0.3135
# RandomForestRegressor 의 정답률:  0.7691
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.691
# RidgeCV 의 정답률:  0.6924
# SGDRegressor 의 정답률:  0.6732
# SVR 의 정답률:  0.5449
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.6702
# TransformedTargetRegressor 의 정답률:  0.6909
# TweedieRegressor 의 정답률:  0.1518
# VotingRegressor 은 안나온 놈

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_diabetes
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = load_diabetes()
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

# ARDRegression 의 정답률:  0.4317
# AdaBoostRegressor 의 정답률:  0.3342
# BaggingRegressor 의 정답률:  0.3186
# BayesianRidge 의 정답률:  0.431
# CCA 의 정답률:  0.4411
# DecisionTreeRegressor 의 정답률:  -0.2863
# DummyRegressor 의 정답률:  -0.0366
# ElasticNet 의 정답률:  0.0877
# ElasticNetCV 의 정답률:  0.4314
# ExtraTreeRegressor 의 정답률:  -0.2629
# ExtraTreesRegressor 의 정답률:  0.3536
# GammaRegressor 의 정답률:  0.0369
# GaussianProcessRegressor 의 정답률:  -8.4939
# GradientBoostingRegressor 의 정답률:  0.311
# HistGradientBoostingRegressor 의 정답률:  0.2625
# HuberRegressor 의 정답률:  0.4322
# IsotonicRegression 은 안나온 놈
# KNeighborsRegressor 의 정답률:  0.3522
# KernelRidge 의 정답률:  0.4274
# Lars 의 정답률:  0.4302
# LarsCV 의 정답률:  0.4193
# Lasso 의 정답률:  0.4286
# LassoCV 의 정답률:  0.4166
# LassoLars 의 정답률:  0.3338
# LassoLarsCV 의 정답률:  0.4165
# LassoLarsIC 의 정답률:  0.4308
# LinearRegression 의 정답률:  0.4302
# LinearSVR 의 정답률:  0.1349
# MLPRegressor 의 정답률:  -0.6284
# MultiOutputRegressor 은 안나온 놈
# MultiTaskElasticNet 은 안나온 놈
# MultiTaskElasticNetCV 은 안나온 놈
# MultiTaskLasso 은 안나온 놈
# MultiTaskLassoCV 은 안나온 놈
# NuSVR 의 정답률:  0.0778
# OrthogonalMatchingPursuit 의 정답률:  0.2676
# OrthogonalMatchingPursuitCV 의 정답률:  0.4223
# PLSCanonical 의 정답률:  -1.3462
# PLSRegression 의 정답률:  0.4284
# PassiveAggressiveRegressor 의 정답률:  0.439
# PoissonRegressor 의 정답률:  0.4263
# RANSACRegressor 의 정답률:  -0.0257
# RadiusNeighborsRegressor 의 정답률:  0.1069
# RandomForestRegressor 의 정답률:  0.3197
# RegressorChain 은 안나온 놈
# Ridge 의 정답률:  0.4328
# RidgeCV 의 정답률:  0.425
# SGDRegressor 의 정답률:  0.4312
# SVR 의 정답률:  0.0882
# StackingRegressor 은 안나온 놈
# TheilSenRegressor 의 정답률:  0.4197
# TransformedTargetRegressor 의 정답률:  0.4302
# TweedieRegressor 의 정답률:  0.0403
# VotingRegressor 은 안나온 놈
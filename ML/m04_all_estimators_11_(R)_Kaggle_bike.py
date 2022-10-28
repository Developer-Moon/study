from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
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


#2. 모델구성
# all_Algorithms = all_estimators(type_filter='classifier') # 분류모델
all_Algorithms = all_estimators(type_filter='regressor')    # 회귀모델
# print(all_Algorithms) 전체 모델 보기
print('모델의 갯수 :', len(all_Algorithms)) # 모델의 갯수 :  41

for (name, algorithms) in all_Algorithms:   # (key, value)
    try:                                    # try 예외처리
        model = algorithms()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률 :', acc)
    except:
        # continue
        print(name, '은 안나온 놈!!!')
        
# 버전에 따라 안돌아가는 모델들이 있다

# 모델의 갯수 : 54
# ARDRegression 의 정답률 : 0.3854032448639464
# AdaBoostRegressor 의 정답률 : 0.6657066863129482
# BaggingRegressor 의 정답률 : 0.9502998459229142
# BayesianRidge 의 정답률 : 0.3866987665481103
# CCA 의 정답률 : 0.1407404585319013
# DecisionTreeRegressor 의 정답률 : 0.9105743640883193
# DummyRegressor 의 정답률 : -0.00028065838954938194
# ElasticNet 의 정답률 : 0.1347244526752397
# ElasticNetCV 의 정답률 : 0.3723873341600379
# ExtraTreeRegressor 의 정답률 : 0.8345241108759164
# ExtraTreesRegressor 의 정답률 : 0.9560770773433372
# GammaRegressor 의 정답률 : 0.08091414905524674
# GaussianProcessRegressor 의 정답률 : -53.823129299221605
# GradientBoostingRegressor 의 정답률 : 0.8719518349794714
# HistGradientBoostingRegressor 의 정답률 : 0.9559994443185389
# HuberRegressor 의 정답률 : 0.3532105820361675
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 : 0.6972545446416499
# KernelRidge 의 정답률 : 0.3864896435904891
# Lars 의 정답률 : 0.3861668147548831
# LarsCV 의 정답률 : 0.38136083436966417
# Lasso 의 정답률 : 0.3828530062151899
# LassoCV 의 정답률 : 0.3861641277107468
# LassoLars 의 정답률 : -0.00028065838954938194
# LassoLarsCV 의 정답률 : 0.3861610238131281
# LassoLarsIC 의 정답률 : 0.3858189010342127
# LinearRegression 의 정답률 : 0.3861668147548829
# LinearSVR 의 정답률 : 0.31691674555042293
# MLPRegressor 의 정답률 : 0.5256162740338213
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 : 0.35230190019955976
# OrthogonalMatchingPursuit 의 정답률 : 0.16434254497329825
# OrthogonalMatchingPursuitCV 의 정답률 : 0.38596346593174113
# PLSCanonical 의 정답률 : -0.29684388801944017
# PLSRegression 의 정답률 : 0.3842643133864829
# PassiveAggressiveRegressor 의 정답률 : 0.34831356083477727
# PoissonRegressor 의 정답률 : 0.3988175242055708
# RANSACRegressor 의 정답률 : 0.13128949325782657
# RadiusNeighborsRegressor 의 정답률 : 0.213900185666289
# RandomForestRegressor 의 정답률 : 0.9553059222402529
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 : 0.38655314869565405
# RidgeCV 의 정답률 : 0.386553148695705
# SGDRegressor 의 정답률 : 0.38744580874390755
# SVR 의 정답률 : 0.33815778687821396
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 : 0.3810488254889577
# TransformedTargetRegressor 의 정답률 : 0.3861668147548829
# TweedieRegressor 의 정답률 : 0.08316045015686702
# VotingRegressor 은 안나온 놈!!!
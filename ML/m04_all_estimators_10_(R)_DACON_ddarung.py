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
# ARDRegression 의 정답률 : 0.6071008550459094
# AdaBoostRegressor 의 정답률 : 0.5630840028119563
# BaggingRegressor 의 정답률 : 0.7734517904912658     
# BayesianRidge 의 정답률 : 0.6079913862467632        
# CCA 의 정답률 : 0.31907765470503247
# DecisionTreeRegressor 의 정답률 : 0.6077923828838125
# DummyRegressor 의 정답률 : -0.0012978150939517707   
# ElasticNet 의 정답률 : 0.5109321545640129
# ElasticNetCV 의 정답률 : 0.6056179235705553      
# ExtraTreeRegressor 의 정답률 : 0.5547715357128404
# ExtraTreesRegressor 의 정답률 : 0.8032957816872986
# GammaRegressor 의 정답률 : 0.4375083468194092     
# GaussianProcessRegressor 의 정답률 : -0.8267979336720026
# GradientBoostingRegressor 의 정답률 : 0.7675075776153988
# HistGradientBoostingRegressor 의 정답률 : 0.7774663541781651
# HuberRegressor 의 정답률 : 0.5926605485167256
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 : 0.6853105015739324
# KernelRidge 의 정답률 : -0.8909914083815802
# Lars 의 정답률 : 0.6074013113540634        
# LarsCV 의 정답률 : 0.6073348203535183
# Lasso 의 정답률 : 0.6068943495763253 
# LassoCV 의 정답률 : 0.6075266112304427
# LassoLars 의 정답률 : 0.2971723334659466       
# LassoLarsCV 의 정답률 : 0.6073348203535183     
# LassoLarsIC 의 정답률 : 0.607348708580151      
# LinearRegression 의 정답률 : 0.6074013113540634
# LinearSVR 의 정답률 : 0.5420899831027852       
# MLPRegressor 의 정답률 : 0.6181331691433731
# MultiOutputRegressor 은 안나온 놈!!!       
# MultiTaskElasticNet 은 안나온 놈!!!        
# MultiTaskElasticNetCV 은 안나온 놈!!!      
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 : 0.4117053808081147
# OrthogonalMatchingPursuit 의 정답률 : 0.41512246128256547 
# OrthogonalMatchingPursuitCV 의 정답률 : 0.5947038724863034
# PLSCanonical 의 정답률 : -0.20230141779601674
# PLSRegression 의 정답률 : 0.6073107355091762
# PassiveAggressiveRegressor 의 정답률 : 0.5524268521706848
# PoissonRegressor 의 정답률 : 0.6724349264685558
# RANSACRegressor 의 정답률 : 0.5303384779293803
# RadiusNeighborsRegressor 은 안나온 놈!!!
# RandomForestRegressor 의 정답률 : 0.7896730303230973
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 : 0.6076251159771311
# RidgeCV 의 정답률 : 0.6076251159770987
# SGDRegressor 의 정답률 : 0.6091171585105035
# SVR 의 정답률 : 0.4204705696270067
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 : 0.6018176572021915
# TransformedTargetRegressor 의 정답률 : 0.6074013113540634
# TweedieRegressor 의 정답률 : 0.43772789353531116
# VotingRegressor 은 안나온 놈!!!
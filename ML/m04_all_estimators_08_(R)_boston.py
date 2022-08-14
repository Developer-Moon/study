from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
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
# ARDRegression 의 정답률 : 0.7099563389977074
# AdaBoostRegressor 의 정답률 : 0.8449151901628851
# BaggingRegressor 의 정답률 : 0.8763064272485204     
# BayesianRidge 의 정답률 : 0.710765845851372
# CCA 의 정답률 : 0.6451896929917802
# DecisionTreeRegressor 의 정답률 : 0.8047582988052118
# DummyRegressor 의 정답률 : -0.05614813414915476     
# ElasticNet 의 정답률 : 0.08346607782844695
# ElasticNetCV 의 정답률 : 0.7050841037275242      
# ExtraTreeRegressor 의 정답률 : 0.8090926153048098
# ExtraTreesRegressor 의 정답률 : 0.8874880054584627     
# GammaRegressor 의 정답률 : 0.09633353981031922
# GaussianProcessRegressor 의 정답률 : -9.944790510034165
# GradientBoostingRegressor 의 정답률 : 0.8716890328835851
# HistGradientBoostingRegressor 의 정답률 : 0.8666726063945716
# HuberRegressor 의 정답률 : 0.627457193082401      
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 : 0.7707422231896033
# KernelRidge 의 정답률 : 0.5541851795653718        
# Lars 의 정답률 : 0.6792829045681439
# LarsCV 의 정답률 : 0.6844710887508514
# Lasso 의 정답률 : 0.17372611223569867
# LassoCV 의 정답률 : 0.7110634121645294    
# LassoLars 의 정답률 : -0.05614813414915476
# LassoLarsCV 의 정답률 : 0.7072492359133302
# LassoLarsIC 의 정답률 : 0.698087181557937      
# LinearRegression 의 정답률 : 0.7128434279973941
# LinearSVR 의 정답률 : 0.5206818714843704       
# MLPRegressor 의 정답률 : 0.3621906585656103
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 : 0.5209836196884117
# OrthogonalMatchingPursuit 의 정답률 : 0.4943292395516461
# OrthogonalMatchingPursuitCV 의 정답률 : 0.6258725698588904
# PLSCanonical 의 정답률 : -1.514316595528638
# PLSRegression 의 정답률 : 0.6606606774157424
# PassiveAggressiveRegressor 의 정답률 : 0.5560277036212846 
# PoissonRegressor 의 정답률 : 0.555932556955798
# RANSACRegressor 의 정답률 : 0.35342485384766353
# RadiusNeighborsRegressor 의 정답률 : 0.31159014123402407
# RandomForestRegressor 의 정답률 : 0.8500206977786874
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 : 0.7030809861383944
# RidgeCV 의 정답률 : 0.7118750468322297
# SGDRegressor 의 정답률 : 0.6636564877516304
# SVR 의 정답률 : 0.5419740294728937
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 : 0.6463209739900769
# TransformedTargetRegressor 의 정답률 : 0.7128434279973941
# TweedieRegressor 의 정답률 : 0.1107833071105383
# VotingRegressor 은 안나온 놈!!!
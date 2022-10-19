from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#

#
#1. 데이터
datasets = load_diabetes()
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
# ARDRegression 의 정답률 : 0.6632102268689342
# AdaBoostRegressor 의 정답률 : 0.6167957469384202
# BaggingRegressor 의 정답률 : 0.4675872296545869
# BayesianRidge 의 정답률 : 0.666326719132324
# CCA 의 정답률 : 0.6551339633289877
# DecisionTreeRegressor 의 정답률 : -0.15943050784437185
# DummyRegressor 의 정답률 : -0.019529888245814142
# ElasticNet 의 정답률 : 0.1653415500549269
# ElasticNetCV 의 정답률 : 0.6591399901802775
# ExtraTreeRegressor 의 정답률 : 0.06901615430714236
# ExtraTreesRegressor 의 정답률 : 0.6677936366875112
# GammaRegressor 의 정답률 : 0.10077998336122296
# GaussianProcessRegressor 의 정답률 : -24.466535157653333
# GradientBoostingRegressor 의 정답률 : 0.6664250780283025
# HistGradientBoostingRegressor 의 정답률 : 0.5824653317402437
# HuberRegressor 의 정답률 : 0.658263198350628
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 : 0.5896856008320275
# KernelRidge 의 정답률 : 0.6683812673588583
# Lars 의 정답률 : 0.65407743684386
# LarsCV 의 정답률 : 0.6651458048572614
# Lasso 의 정답률 : 0.642854067192079
# LassoCV 의 정답률 : 0.6590046787657708
# LassoLars 의 정답률 : 0.4595945627634165
# LassoLarsCV 의 정답률 : 0.6603706561649021
# LassoLarsIC 의 정답률 : 0.6658261304900683
# LinearRegression 의 정답률 : 0.6557534150889774
# LinearSVR 의 정답률 : 0.43773232370138593
# MLPRegressor 의 정답률 : -0.5062291554307081
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 : 0.21431773004275012
# OrthogonalMatchingPursuit 의 정답률 : 0.4089663499722199
# OrthogonalMatchingPursuitCV 의 정답률 : 0.6503620856899237
# PLSCanonical 의 정답률 : -1.3493014516751844
# PLSRegression 의 정답률 : 0.6583067738400631
# PassiveAggressiveRegressor 의 정답률 : 0.6423905928436927
# PoissonRegressor 의 정답률 : 0.6740696939876487
# RANSACRegressor 의 정답률 : 0.2588967274289393
# RadiusNeighborsRegressor 의 정답률 : 0.21251799421852002
# RandomForestRegressor 의 정답률 : 0.5945902737883975
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 : 0.6661319024579306
# RidgeCV 의 정답률 : 0.6635155777195565
# SGDRegressor 의 정답률 : 0.6663917955223964
# SVR 의 정답률 : 0.2400781475710959
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 : 0.671871464879899
# TransformedTargetRegressor 의 정답률 : 0.6557534150889774
# TweedieRegressor 의 정답률 : 0.0960210598710225
# VotingRegressor 은 안나온 놈!!!
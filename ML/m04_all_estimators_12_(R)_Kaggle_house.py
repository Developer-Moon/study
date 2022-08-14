from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from tqdm import tqdm_notebook
import pandas as pd
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
# ARDRegression 의 정답률 : 0.84160637076675
# AdaBoostRegressor 의 정답률 : 0.7774558178812359
# BaggingRegressor 의 정답률 : 0.8798254770549804
# BayesianRidge 의 정답률 : 0.8467026842227725
# CCA 의 정답률 : -0.6622184185052074
# DecisionTreeRegressor 의 정답률 : 0.746021973015816
# DummyRegressor 의 정답률 : -0.005088694107215019
# ElasticNet 의 정답률 : 0.8560720267292395
# ElasticNetCV 의 정답률 : 0.6371225278888457
# ExtraTreeRegressor 의 정답률 : 0.7170730142183421
# ExtraTreesRegressor 의 정답률 : 0.8770803724369443
# GammaRegressor 의 정답률 : -0.005088694107215019
# GaussianProcessRegressor 의 정답률 : -6.089742132226499
# GradientBoostingRegressor 의 정답률 : 0.9065819107000205
# HistGradientBoostingRegressor 의 정답률 : 0.8810032840943991
# HuberRegressor 의 정답률 : 0.7638479947784257
# IsotonicRegression 은 안나온 놈!!!
# KNeighborsRegressor 의 정답률 : 0.5677054433046803
# KernelRidge 의 정답률 : 0.8374624782059419
# Lars 의 정답률 : -19.27375163543573
# LarsCV 의 정답률 : 0.8251328971661835
# Lasso 의 정답률 : 0.8373296618602561
# LassoCV 의 정답률 : 0.7838258636695332
# LassoLars 의 정답률 : 0.8386087263887878
# LassoLarsCV 의 정답률 : 0.8471529953494394
# LassoLarsIC 의 정답률 : 0.8436429595369104
# LinearRegression 의 정답률 : 0.8372976697328177
# LinearSVR 의 정답률 : 0.6482273006460944
# MLPRegressor 의 정답률 : 0.6877323886818361
# MultiOutputRegressor 은 안나온 놈!!!
# MultiTaskElasticNet 은 안나온 놈!!!
# MultiTaskElasticNetCV 은 안나온 놈!!!
# MultiTaskLasso 은 안나온 놈!!!
# MultiTaskLassoCV 은 안나온 놈!!!
# NuSVR 의 정답률 : -0.0038600233284815655
# OrthogonalMatchingPursuit 의 정답률 : 0.8144747814396575
# OrthogonalMatchingPursuitCV 의 정답률 : 0.8144747814396575
# PLSCanonical 의 정답률 : -6.176254801585236
# PLSRegression 의 정답률 : 0.8460923593074949
# PassiveAggressiveRegressor 의 정답률 : -0.3559906531258812
# PoissonRegressor 의 정답률 : -0.005088694107215019
# RANSACRegressor 의 정답률 : 0.7839416473302255
# RadiusNeighborsRegressor 의 정답률 : -1.6501191080797437e+28
# RandomForestRegressor 의 정답률 : 0.8888188466744793
# RegressorChain 은 안나온 놈!!!
# Ridge 의 정답률 : 0.8372381232157003
# RidgeCV 의 정답률 : 0.8382436263528801
# SGDRegressor 의 정답률 : -1.7592501040517236e+23
# SVR 의 정답률 : -0.03824031278667439
# StackingRegressor 은 안나온 놈!!!
# TheilSenRegressor 의 정답률 : 0.8730654859391578
# TransformedTargetRegressor 의 정답률 : 0.8372976697328177
# TweedieRegressor 의 정답률 : 0.7564160242793736
# VotingRegressor 은 안나온 놈!!!
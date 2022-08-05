from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
import sklearn as sk
print(sk.__version__)             # 0.24.2
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5 # n_splits=5 5등분
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66 )  

#2. 모델구성
# all_Algorithms = all_estimators(type_filter='classifier') # 분류모델 
all_Algorithms = all_estimators(type_filter='regressor')  # 회귀모델 
# print(all_Algorithms) 전체 모델 보기
print('모델의 갯수 : ', len(all_Algorithms)) # 모델의 갯수 :  41

for (name, algorithms) in all_Algorithms:   # (key, value)
    try:                                    # try 예외처리
        model = algorithms()
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률 :', round(np.mean(score), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')
         
"""
모델의 갯수 :  54
ARDRegression 의 정답률 : 0.7117
AdaBoostRegressor 의 정답률 : 0.787
BaggingRegressor 의 정답률 : 0.8338
BayesianRidge 의 정답률 : 0.7137
CCA 의 정답률 : 0.6339
DecisionTreeRegressor 의 정답률 : 0.7246
DummyRegressor 의 정답률 : -0.018
ElasticNet 의 정답률 : 0.1338
ElasticNetCV 의 정답률 : 0.7137
ExtraTreeRegressor 의 정답률 : 0.7255
ExtraTreesRegressor 의 정답률 : 0.8575
GammaRegressor 의 정답률 : 0.1727
GaussianProcessRegressor 의 정답률 : -1.4277
GradientBoostingRegressor 의 정답률 : 0.8679
HistGradientBoostingRegressor 의 정답률 : 0.818
HuberRegressor 의 정답률 : 0.7069
IsotonicRegression 은 안나온 놈!!!    
KNeighborsRegressor 의 정답률 : 0.6956
KernelRidge 의 정답률 : 0.6292
Lars 의 정답률 : 0.701
LarsCV 의 정답률 : 0.692
Lasso 의 정답률 : 0.2054
LassoCV 의 정답률 : 0.7119  
LassoLars 의 정답률 : -0.018
LassoLarsCV 의 정답률 : 0.7124
LassoLarsIC 의 정답률 : 0.7028
LinearRegression 의 정답률 : 0.712
LinearSVR 의 정답률 : 0.6081
MLPRegressor 의 정답률 : 0.1106     
MultiOutputRegressor 은 안나온 놈!!!
MultiTaskElasticNet 은 안나온 놈!!! 
MultiTaskElasticNetCV 은 안나온 놈!!!
MultiTaskLasso 은 안나온 놈!!!       
MultiTaskLassoCV 은 안나온 놈!!!     
NuSVR 의 정답률 : 0.5523
OrthogonalMatchingPursuit 의 정답률 : 0.5381
OrthogonalMatchingPursuitCV 의 정답률 : 0.6752
PLSCanonical 의 정답률 : -2.3326
PLSRegression 의 정답률 : 0.6964
PassiveAggressiveRegressor 의 정답률 : 0.6528
PoissonRegressor 의 정답률 : 0.6204
RANSACRegressor 의 정답률 : -0.179
RadiusNeighborsRegressor 은 안나온 놈!!!
RandomForestRegressor 의 정답률 : 0.8307
RegressorChain 은 안나온 놈!!!
Ridge 의 정답률 : 0.713
RidgeCV 의 정답률 : 0.7093
SGDRegressor 의 정답률 : 0.6981
SVR 의 정답률 : 0.5687
StackingRegressor 은 안나온 놈!!!
TheilSenRegressor 의 정답률 : 0.698
TransformedTargetRegressor 의 정답률 : 0.712
TweedieRegressor 의 정답률 : 0.1648
VotingRegressor 은 안나온 놈!!!
"""        
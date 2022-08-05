from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, AbsScaler 
from sklearn.datasets import fetch_california_housing
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
datasets = fetch_california_housing()
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
ARDRegression 의 정답률 : 0.61
AdaBoostRegressor 의 정답률 : 0.4435
BaggingRegressor 의 정답률 : 0.7835
BayesianRidge 의 정답률 : 0.6099
CCA 의 정답률 : 0.5653
DecisionTreeRegressor 의 정답률 : 0.6179
DummyRegressor 의 정답률 : -0.0006
ElasticNet 의 정답률 : -0.0006
ElasticNetCV 의 정답률 : 0.6013
ExtraTreeRegressor 의 정답률 : 0.5516
ExtraTreesRegressor 의 정답률 : 0.8094
GammaRegressor 의 정답률 : 0.0187
GaussianProcessRegressor 의 정답률 : -13068.1396
GradientBoostingRegressor 의 정답률 : 0.7848
HistGradientBoostingRegressor 의 정답률 : 0.8329
HuberRegressor 의 정답률 : 0.5768
IsotonicRegression 은 안나온 놈!!!
KNeighborsRegressor 의 정답률 : 0.6994
KernelRidge 의 정답률 : 0.5339
Lars 의 정답률 : 0.6099
LarsCV 의 정답률 : 0.6093
Lasso 의 정답률 : -0.0006
LassoCV 의 정답률 : 0.6088
LassoLars 의 정답률 : -0.0006
LassoLarsCV 의 정답률 : 0.6096
LassoLarsIC 의 정답률 : 0.6099
LinearRegression 의 정답률 : 0.6099
LinearSVR 의 정답률 : 0.5867
MLPRegressor 의 정답률 : 0.7252
MultiOutputRegressor 은 안나온 놈!!!
MultiTaskElasticNet 은 안나온 놈!!!
MultiTaskElasticNetCV 은 안나온 놈!!!
MultiTaskLasso 은 안나온 놈!!!
MultiTaskLassoCV 은 안나온 놈!!!
NuSVR 의 정답률 : 0.6648
OrthogonalMatchingPursuit 의 정답률 : 0.4752
OrthogonalMatchingPursuitCV 의 정답률 : 0.5989
PLSCanonical 의 정답률 : 0.3707
PLSRegression 의 정답률 : 0.523
PassiveAggressiveRegressor 의 정답률 : 0.2258
PoissonRegressor 의 정답률 : 0.0407
RANSACRegressor 의 정답률 : -2.8665
RadiusNeighborsRegressor 은 안나온 놈!!!
RandomForestRegressor 의 정답률 : 0.8069
RegressorChain 은 안나온 놈!!!
Ridge 의 정답률 : 0.6033
RidgeCV 의 정답률 : 0.6069
SGDRegressor 의 정답률 : 0.5633
SVR 의 정답률 : 0.6613
StackingRegressor 은 안나온 놈!!!
TheilSenRegressor 의 정답률 : -2.9958
TransformedTargetRegressor 의 정답률 : 0.6099
TweedieRegressor 의 정답률 : 0.0188
VotingRegressor 은 안나온 놈!!!
"""        
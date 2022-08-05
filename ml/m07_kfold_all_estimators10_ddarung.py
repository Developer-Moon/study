from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
import numpy as np
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
import sklearn as sk
print(sk.__version__)             # 0.24.2
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


# 1. 데이터
path = './_data/ddarung/'                                         # path(변수)에 경로를 넣음
train_set = pd.read_csv(path + 'train.csv', index_col=0)          # 판다스로 csv(엑셀시트)파일을 읽어라   path(경로) + train.csv                                                               
test_set = pd.read_csv(path + 'test.csv', index_col=0)            # 이 값은 예측 부분에서 쓴다   

print(test_set.shape) # (715, 9)

train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   # fillna() - 결측값을 (특정값)로 채우겠다
                                              # 결측값을 결측값의 앞 행의 값으로 채우기 : df.fillna(method='ffill') or df.fillna(method='pad')
                                              # 결측값을 결측값의 뒷 행의 값으로 채우기 : df.fillna(method='bfill') or df.fillna(method='backfill')
                                              # 결측값을 각 열의 평균 값으로 채우기     : df.fillna(df.mean())
                                              
print(train_set.isnull().sum())               # train 결측지 평균값으로 채움                                     
print(test_set.isnull().sum())                # test 결측지 평균값으로 채움      
    
                                             
train_set = train_set.dropna()                # dropna() - 행별로 싹 날려뿌겠다 : 결측지를 제거 하는 법[위 에서 결측지를 채워서 지금은 의미 없다]
                                              # 결측값 있는 행 제거 : df.dropna() or df.dropna(axis=0)
                                              # 결측값 있는 열 제거 : df.dropna(axis=1)

x = train_set.drop(['count'], axis=1)         # train_set에서 count를 drop(뺀다) axis=1 열, axis=0 행 
print(x)
print(x.columns)                              # [1459 rows x 9 columns]
print(x.shape)                                # (1459, 9) - input_dim=9

y = train_set['count']                        # y는 train_set에서 count컬럼이다
print(y)  
print(y.shape)                                # (1459,) 1459개의 스칼라  output=1    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=16)

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
ARDRegression 의 정답률 : 0.5758
AdaBoostRegressor 의 정답률 : 0.6256
BaggingRegressor 의 정답률 : 0.7559
BayesianRidge 의 정답률 : 0.5767
CCA 의 정답률 : 0.2148
DecisionTreeRegressor 의 정답률 : 0.5488
DummyRegressor 의 정답률 : -0.0018
ElasticNet 의 정답률 : 0.2067
ElasticNetCV 의 정답률 : 0.5646
ExtraTreeRegressor 의 정답률 : 0.5439
ExtraTreesRegressor 의 정답률 : 0.7848
GammaRegressor 의 정답률 : 0.0975
GaussianProcessRegressor 의 정답률 : -37.3415
GradientBoostingRegressor 의 정답률 : 0.7695
HistGradientBoostingRegressor 의 정답률 : 0.7832
HuberRegressor 의 정답률 : 0.5653
IsotonicRegression 은 안나온 놈!!!
KNeighborsRegressor 의 정답률 : 0.6996
KernelRidge 의 정답률 : 0.5758
Lars 의 정답률 : 0.5767
LarsCV 의 정답률 : 0.5735
Lasso 의 정답률 : 0.562
LassoCV 의 정답률 : 0.5763
LassoLars 의 정답률 : 0.3588
LassoLarsCV 의 정답률 : 0.5748
LassoLarsIC 의 정답률 : 0.5772
LinearRegression 의 정답률 : 0.5767
LinearSVR 의 정답률 : 0.4517
MLPRegressor 의 정답률 : 0.4446
MultiOutputRegressor 은 안나온 놈!!!
MultiTaskElasticNet 은 안나온 놈!!!
MultiTaskElasticNetCV 은 안나온 놈!!!
MultiTaskLasso 은 안나온 놈!!!
MultiTaskLassoCV 은 안나온 놈!!!
NuSVR 의 정답률 : 0.4454
OrthogonalMatchingPursuit 의 정답률 : 0.3673
OrthogonalMatchingPursuitCV 의 정답률 : 0.5691
PLSCanonical 의 정답률 : -0.4681
PLSRegression 의 정답률 : 0.5668
PassiveAggressiveRegressor 의 정답률 : 0.5511
PoissonRegressor 의 정답률 : 0.5984
RANSACRegressor 의 정답률 : 0.4502
RadiusNeighborsRegressor 의 정답률 : 0.2985
RandomForestRegressor 의 정답률 : 0.7759
RegressorChain 은 안나온 놈!!!
Ridge 의 정답률 : 0.5769
RidgeCV 의 정답률 : 0.5769
SGDRegressor 의 정답률 : 0.5743
SVR 의 정답률 : 0.4421
StackingRegressor 은 안나온 놈!!!
TheilSenRegressor 의 정답률 : 0.562
TransformedTargetRegressor 의 정답률 : 0.5767
TweedieRegressor 의 정답률 : 0.1288
VotingRegressor 은 안나온 놈!!!    
"""        
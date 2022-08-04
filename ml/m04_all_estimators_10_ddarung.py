from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
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



#2. 모델구성
# all_Algorithms = all_estimators(type_filter='classifier') # 분류모델
all_Algorithms = all_estimators(type_filter='regressor')  # 회귀모델
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
        

"""
모델의 갯수 : 54
ARDRegression 의 정답률 : 0.6374364209200268
AdaBoostRegressor 의 정답률 : 0.6457083790437292
BaggingRegressor 의 정답률 : 0.7505333350511276
BayesianRidge 의 정답률 : 0.6307824338673642
CCA 의 정답률 : 0.32468379754300125
DecisionTreeRegressor 의 정답률 : 0.5494152369751477
DummyRegressor 의 정답률 : -0.0004804249939593941
ElasticNet 의 정답률 : 0.6191685189594502
ElasticNetCV 의 정답률 : 0.5773992094209466
ExtraTreeRegressor 의 정답률 : 0.5276428906900907
ExtraTreesRegressor 의 정답률 : 0.7717167112968653
GammaRegressor 의 정답률 : -0.0004804249939593941
GaussianProcessRegressor 의 정답률 : -1.7271466992151794
GradientBoostingRegressor 의 정답률 : 0.7475460441880073
HistGradientBoostingRegressor 의 정답률 : 0.7777882621518821
HuberRegressor 의 정답률 : 0.5975420220763303
IsotonicRegression 은 안나온 놈!!!
KNeighborsRegressor 의 정답률 : 0.4331631291715098
KernelRidge 의 정답률 : 0.6366533638544749
Lars 의 정답률 : 0.6366061065665973
LarsCV 의 정답률 : 0.6338422364711889
Lasso 의 정답률 : 0.6257613430341391
LassoCV 의 정답률 : 0.6081680508355227
LassoLars 의 정답률 : 0.3264303534189422
LassoLarsCV 의 정답률 : 0.6338422364711889
LassoLarsIC 의 정답률 : 0.6347293634149473
LinearRegression 의 정답률 : 0.6366061065665973
LinearSVR 의 정답률 : -0.26685846581107975
MLPRegressor 의 정답률 : 0.626709604367613
MultiOutputRegressor 은 안나온 놈!!!
MultiTaskElasticNet 은 안나온 놈!!!
MultiTaskElasticNetCV 은 안나온 놈!!!
MultiTaskLasso 은 안나온 놈!!!
MultiTaskLassoCV 은 안나온 놈!!!
NuSVR 의 정답률 : 0.04922343294366793
OrthogonalMatchingPursuit 의 정답률 : 0.4049590061417633
OrthogonalMatchingPursuitCV 의 정답률 : 0.6147710470297965
PLSCanonical 의 정답률 : -0.27046627963853487
PLSRegression 의 정답률 : 0.6325324821340439
PassiveAggressiveRegressor 의 정답률 : 0.32874026709663307
PoissonRegressor 의 정답률 : -0.0004804249939593941
RANSACRegressor 의 정답률 : 0.5543199200487142
RadiusNeighborsRegressor 은 안나온 놈!!!
RandomForestRegressor 의 정답률 : 0.7650808219302678
RegressorChain 은 안나온 놈!!!
Ridge 의 정답률 : 0.6329487433389992
RidgeCV 의 정답률 : 0.6356719546684637
SGDRegressor 의 정답률 : -5.366824727613083e+25
SVR 의 정답률 : 0.050106415096281554
StackingRegressor 은 안나온 놈!!!
TheilSenRegressor 의 정답률 : 0.6215248148194797
TransformedTargetRegressor 의 정답률 : 0.6366061065665973
TweedieRegressor 의 정답률 : 0.6165412737497936
VotingRegressor 은 안나온 놈!!!
"""        
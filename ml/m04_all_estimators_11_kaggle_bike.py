from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
path = './_data/kaggle_bike/'        
train_set = pd.read_csv(path + 'train.csv', index_col=0)   
test_set = pd.read_csv(path + 'test.csv', index_col=0)  

x = train_set.drop(['casual', 'registered', 'count'], axis=1)  
y = train_set['count']   
  
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75)  
    
scaler = RobustScaler()
scaler.fit(x_train)
scaler.fit(test_set)
test_set = scaler.transform(test_set)
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
        
"""
모델의 갯수 : 54
ARDRegression 의 정답률 : 0.25154194439325595
AdaBoostRegressor 의 정답률 : 0.18412275806060718
BaggingRegressor 의 정답률 : 0.24144228100186704
BayesianRidge 의 정답률 : 0.25112458709842544
CCA 의 정답률 : -0.21769772597362835
DecisionTreeRegressor 의 정답률 : -0.1352385960226412
DummyRegressor 의 정답률 : -0.001591867190416263
ElasticNet 의 정답률 : 0.2066089074004288
ElasticNetCV 의 정답률 : 0.2521035780972485
ExtraTreeRegressor 의 정답률 : -0.110159643756373
ExtraTreesRegressor 의 정답률 : 0.19667285055938744
GammaRegressor 의 정답률 : 0.16771307063298702
GaussianProcessRegressor 의 정답률 : -28063.444893167267
GradientBoostingRegressor 의 정답률 : 0.32793028685726633
HistGradientBoostingRegressor 의 정답률 : 0.34887344939805887
HuberRegressor 의 정답률 : 0.2311390217848509
IsotonicRegression 은 안나온 놈!!!
KNeighborsRegressor 의 정답률 : 0.2912359499354492
KernelRidge 의 정답률 : -0.31079000600555773
Lars 의 정답률 : 0.2511017545946507
LarsCV 의 정답률 : 0.2517701915243975
Lasso 의 정답률 : 0.2524323989512268
LassoCV 의 정답률 : 0.2517981592498967
LassoLars 의 정답률 : -0.001591867190416263
LassoLarsCV 의 정답률 : 0.2517701915243975
LassoLarsIC 의 정답률 : 0.2519022674992113
LinearRegression 의 정답률 : 0.2511017545946507
LinearSVR 의 정답률 : 0.20642112760243536
MLPRegressor 의 정답률 : 0.301134383074531
MultiOutputRegressor 은 안나온 놈!!!
MultiTaskElasticNet 은 안나온 놈!!!
MultiTaskElasticNetCV 은 안나온 놈!!!
MultiTaskLasso 은 안나온 놈!!!
MultiTaskLassoCV 은 안나온 놈!!!
NuSVR 의 정답률 : 0.23712840250854206
OrthogonalMatchingPursuit 의 정답률 : 0.15223553320445116
OrthogonalMatchingPursuitCV 의 정답률 : 0.24763796798541793
PLSCanonical 의 정답률 : -0.6506115489126874
PLSRegression 의 정답률 : 0.24853179038621498
PassiveAggressiveRegressor 의 정답률 : 0.17841290305886692
PoissonRegressor 의 정답률 : 0.24811314385919236
RANSACRegressor 의 정답률 : 0.019330446108704935
RadiusNeighborsRegressor 의 정답률 : -3.000360922039419e+30
RandomForestRegressor 의 정답률 : 0.3044879729612965
RegressorChain 은 안나온 놈!!!
Ridge 의 정답률 : 0.25110210793895393
RidgeCV 의 정답률 : 0.251120354349064
SGDRegressor 의 정답률 : 0.2493237897710605
SVR 의 정답률 : 0.22697812498136538
StackingRegressor 은 안나온 놈!!!  
TheilSenRegressor 의 정답률 : 0.24445211799864974        
TransformedTargetRegressor 의 정답률 : 0.2511017545946507
TweedieRegressor 의 정답률 : 0.16631090821895067
VotingRegressor 은 안나온 놈!!!
"""        
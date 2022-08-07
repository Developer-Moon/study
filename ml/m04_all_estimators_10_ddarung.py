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
path = './_data/ddarung/'                                       
train_set = pd.read_csv(path + 'train.csv', index_col=0)                                                               
test_set = pd.read_csv(path + 'test.csv', index_col=0) 

train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   
                                                                                           
train_set = train_set.dropna() 

x = train_set.drop(['count'], axis=1)        
y = train_set['count']                  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=16)

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, accuracy_score
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
import sklearn as sk
print(sk.__version__)             # 0.24.2
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


# 1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



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
ARDRegression 의 정답률 : 0.6304901603851405
AdaBoostRegressor 의 정답률 : 0.4614953199702513
BaggingRegressor 의 정답률 : 0.7901761335882121
BayesianRidge 의 정답률 : 0.6303910972564213
CCA 의 정답률 : 0.589507527501187
DecisionTreeRegressor 의 정답률 : 0.6147337324181772
DummyRegressor 의 정답률 : -4.883634526553493e-06
ElasticNet 의 정답률 : -4.883634526553493e-06
ElasticNetCV 의 정답률 : 0.6206511074459923
ExtraTreeRegressor 의 정답률 : 0.5865502921813046
ExtraTreesRegressor 의 정답률 : 0.8260712211249341
GammaRegressor 의 정답률 : 0.019988259524611918
GaussianProcessRegressor 의 정답률 : -9361.488029447477
GradientBoostingRegressor 의 정답률 : 0.7979257462738886
HistGradientBoostingRegressor 의 정답률 : 0.8374084146143119
HuberRegressor 의 정답률 : 0.5977120077373659
IsotonicRegression 은 안나온 놈!!!
KNeighborsRegressor 의 정답률 : 0.7129214803137134
KernelRidge 의 정답률 : 0.547090847490874
Lars 의 정답률 : 0.6304493823301508      
LarsCV 의 정답률 : 0.6298679705725403   
Lasso 의 정답률 : -4.883634526553493e-06
LassoCV 의 정답률 : 0.6289682344750812      
LassoLars 의 정답률 : -4.883634526553493e-06
LassoLarsCV 의 정답률 : 0.6298679705725403     
LassoLarsIC 의 정답률 : 0.6304493823301508     
LinearRegression 의 정답률 : 0.6304493823301511
LinearSVR 의 정답률 : 0.6109107568859149
MLPRegressor 의 정답률 : 0.7397849119095643
MultiOutputRegressor 은 안나온 놈!!!       
MultiTaskElasticNet 은 안나온 놈!!!        
MultiTaskElasticNetCV 은 안나온 놈!!!      
MultiTaskLasso 은 안나온 놈!!!
MultiTaskLassoCV 은 안나온 놈!!!
NuSVR 의 정답률 : 0.6799232281830356
OrthogonalMatchingPursuit 의 정답률 : 0.4959432263287289
OrthogonalMatchingPursuitCV 의 정답률 : 0.6236693745707413
PLSCanonical 의 정답률 : 0.40306829276739664
PLSRegression 의 정답률 : 0.5441430665768268
PassiveAggressiveRegressor 의 정답률 : 0.33780473775482867
PoissonRegressor 의 정답률 : 0.040600428730287685
RANSACRegressor 의 정답률 : -1.637546928022783
RadiusNeighborsRegressor 의 정답률 : 0.014872119100431624
RandomForestRegressor 의 정답률 : 0.8158945419066311
RegressorChain 은 안나온 놈!!!
Ridge 의 정답률 : 0.6235491010242271
RidgeCV 의 정답률 : 0.6292550311329156
SGDRegressor 의 정답률 : 0.5847300105596819
SVR 의 정답률 : 0.6776238899656893
StackingRegressor 은 안나온 놈!!!
TheilSenRegressor 의 정답률 : -16.320602879536693
TransformedTargetRegressor 의 정답률 : 0.6304493823301511
TweedieRegressor 의 정답률 : 0.020099802295124536
VotingRegressor 은 안나온 놈!!!
"""        
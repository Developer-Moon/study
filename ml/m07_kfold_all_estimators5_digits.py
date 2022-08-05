from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
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
datasets = load_digits()
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
all_Algorithms = all_estimators(type_filter='classifier') # 분류모델 
# all_Algorithms = all_estimators(type_filter='regressor')  # 회귀모델 
# print(all_Algorithms) 전체 모델 보기
print('모델의 갯수 : ', len(all_Algorithms)) # 모델의 갯수 :  41

for (name, algorithms) in all_Algorithms:   # (key, value)
    try:                                    # try 예외처리
        model = algorithms()
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        
        y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 :', round(np.mean(score), 4))
    except:
        # continue
        print(name, '은 안나온 놈!!!')
         
"""
모델의 갯수 :  41
AdaBoostClassifier 의 정답률 : 0.3305
BaggingClassifier 의 정답률 : 0.9165
BernoulliNB 의 정답률 : 0.8567
CalibratedClassifierCV 의 정답률 : 0.9596
CategoricalNB 은 안나온 놈!!!
ClassifierChain 은 안나온 놈!!!
ComplementNB 의 정답률 : 0.8205
DecisionTreeClassifier 의 정답률 : 0.8518
DummyClassifier 의 정답률 : 0.0898
ExtraTreeClassifier 의 정답률 : 0.7822
ExtraTreesClassifier 의 정답률 : 0.977
GaussianNB 의 정답률 : 0.8567
GaussianProcessClassifier 의 정답률 : 0.9819
GradientBoostingClassifier 의 정답률 : 0.9631
HistGradientBoostingClassifier 의 정답률 : 0.9701
KNeighborsClassifier 의 정답률 : 0.9826
LabelPropagation 의 정답률 : 0.9875
LabelSpreading 의 정답률 : 0.9875
LinearDiscriminantAnalysis 의 정답률 : 0.9534
LinearSVC 의 정답률 : 0.9603
LogisticRegression 의 정답률 : 0.961
LogisticRegressionCV 의 정답률 : 0.9631
MLPClassifier 의 정답률 : 0.9729
MultiOutputClassifier 은 안나온 놈!!!
MultinomialNB 의 정답률 : 0.9033
NearestCentroid 의 정답률 : 0.9047
NuSVC 의 정답률 : 0.9631
OneVsOneClassifier 은 안나온 놈!!!
OneVsRestClassifier 은 안나온 놈!!!
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.9471
Perceptron 의 정답률 : 0.9464
QuadraticDiscriminantAnalysis 의 정답률 : 0.8741
RadiusNeighborsClassifier 은 안나온 놈!!!
RandomForestClassifier 의 정답률 : 0.9673
RidgeClassifier 의 정답률 : 0.9276
RidgeClassifierCV 의 정답률 : 0.9276
SGDClassifier 의 정답률 : 0.9464
SVC 의 정답률 : 0.9847
StackingClassifier 은 안나온 놈!!!
VotingClassifier 은 안나온 놈!!!
"""        
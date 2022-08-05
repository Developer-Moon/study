from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
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
datasets = load_wine()
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
AdaBoostClassifier 의 정답률 : 0.8303
BaggingClassifier 의 정답률 : 0.9291
BernoulliNB 의 정답률 : 0.4091
CalibratedClassifierCV 의 정답률 : 0.9857
CategoricalNB 은 안나온 놈!!!  
ClassifierChain 은 안나온 놈!!!
ComplementNB 의 정답률 : 0.8732
DecisionTreeClassifier 의 정답률 : 0.9145
DummyClassifier 의 정답률 : 0.4232       
ExtraTreeClassifier 의 정답률 : 0.8877   
ExtraTreesClassifier 의 정답률 : 0.986
GaussianNB 의 정답률 : 0.9857
GaussianProcessClassifier 의 정답률 : 0.9717
GradientBoostingClassifier 의 정답률 : 0.9153
HistGradientBoostingClassifier 의 정답률 : 0.936
KNeighborsClassifier 의 정답률 : 0.9507
LabelPropagation 의 정답률 : 0.9438
LabelSpreading 의 정답률 : 0.9438
LinearDiscriminantAnalysis 의 정답률 : 0.986
LinearSVC 의 정답률 : 0.9786
LogisticRegression 의 정답률 : 0.9788
LogisticRegressionCV 의 정답률 : 0.986
MLPClassifier 의 정답률 : 0.9574
MultiOutputClassifier 은 안나온 놈!!!
MultinomialNB 의 정답률 : 0.8943
NearestCentroid 의 정답률 : 0.9507
NuSVC 은 안나온 놈!!!
OneVsOneClassifier 은 안나온 놈!!!
OneVsRestClassifier 은 안나온 놈!!!
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.9786
Perceptron 의 정답률 : 0.9788
QuadraticDiscriminantAnalysis 의 정답률 : 0.9857
RadiusNeighborsClassifier 은 안나온 놈!!!
RandomForestClassifier 의 정답률 : 0.9788
RidgeClassifier 의 정답률 : 0.9929
RidgeClassifierCV 의 정답률 : 0.986
SGDClassifier 의 정답률 : 0.9857
SVC 의 정답률 : 0.9929
StackingClassifier 은 안나온 놈!!!
VotingClassifier 은 안나온 놈!!!
"""        
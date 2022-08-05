from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer
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
datasets = load_breast_cancer()
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
AdaBoostClassifier 의 정답률 : 0.9582
BaggingClassifier 의 정답률 : 0.9429
BernoulliNB 의 정답률 : 0.611
CalibratedClassifierCV 의 정답률 : 0.9736
CategoricalNB 은 안나온 놈!!!
ClassifierChain 은 안나온 놈!!!
ComplementNB 은 안나온 놈!!!
DecisionTreeClassifier 의 정답률 : 0.9121
DummyClassifier 의 정답률 : 0.622
ExtraTreeClassifier 의 정답률 : 0.9187
ExtraTreesClassifier 의 정답률 : 0.9648
GaussianNB 의 정답률 : 0.9319
GaussianProcessClassifier 의 정답률 : 0.9604
GradientBoostingClassifier 의 정답률 : 0.9538
HistGradientBoostingClassifier 의 정답률 : 0.956
KNeighborsClassifier 의 정답률 : 0.967
LabelPropagation 의 정답률 : 0.9692
LabelSpreading 의 정답률 : 0.967
LinearDiscriminantAnalysis 의 정답률 : 0.956
LinearSVC 의 정답률 : 0.9758
LogisticRegression 의 정답률 : 0.9626
LogisticRegressionCV 의 정답률 : 0.9758
MLPClassifier 의 정답률 : 0.967      
MultiOutputClassifier 은 안나온 놈!!!
MultinomialNB 은 안나온 놈!!!        
NearestCentroid 의 정답률 : 0.9385
NuSVC 의 정답률 : 0.9451
OneVsOneClassifier 은 안나온 놈!!!  
OneVsRestClassifier 은 안나온 놈!!! 
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.9626
Perceptron 의 정답률 : 0.956
QuadraticDiscriminantAnalysis 의 정답률 : 0.9495
RadiusNeighborsClassifier 의 정답률 : nan
RandomForestClassifier 의 정답률 : 0.9582
RidgeClassifier 의 정답률 : 0.9582
RidgeClassifierCV 의 정답률 : 0.9604
SGDClassifier 의 정답률 : 0.9736
SVC 의 정답률 : 0.978
StackingClassifier 은 안나온 놈!!!
VotingClassifier 은 안나온 놈!!!  
"""        
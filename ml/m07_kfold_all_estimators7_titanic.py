from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_covtype
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
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

print(train_set.describe())
print(train_set.info())
print(train_set.isnull())
print(train_set.isnull().sum())
print(train_set.shape) # (10886, 12)
print(train_set.columns.values) # ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
print(train_set['Embarked'].mode())
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
print(test_set.isnull().sum())
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
print(test_set['Embarked'].mode())
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) 
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환

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
AdaBoostClassifier 의 정답률 : 0.8034
BaggingClassifier 의 정답률 : 0.8188
BernoulliNB 의 정답률 : 0.7907
CalibratedClassifierCV 의 정답률 : 0.8048
CategoricalNB 은 안나온 놈!!!
ClassifierChain 은 안나온 놈!!!
ComplementNB 의 정답률 : 0.7837
DecisionTreeClassifier 의 정답률 : 0.7823
DummyClassifier 의 정답률 : 0.6306
ExtraTreeClassifier 의 정답률 : 0.7654
ExtraTreesClassifier 의 정답률 : 0.816
GaussianNB 의 정답률 : 0.8005
GaussianProcessClassifier 의 정답률 : 0.823
GradientBoostingClassifier 의 정답률 : 0.8343
HistGradientBoostingClassifier 의 정답률 : 0.8356
KNeighborsClassifier 의 정답률 : 0.8146
LabelPropagation 의 정답률 : 0.8132
LabelSpreading 의 정답률 : 0.8146
LinearDiscriminantAnalysis 의 정답률 : 0.802
LinearSVC 의 정답률 : 0.802
LogisticRegression 의 정답률 : 0.8076
LogisticRegressionCV 의 정답률 : 0.8048
MLPClassifier 의 정답률 : 0.8174     
MultiOutputClassifier 은 안나온 놈!!!
MultinomialNB 의 정답률 : 0.8034     
NearestCentroid 의 정답률 : 0.7935
NuSVC 의 정답률 : 0.823
OneVsOneClassifier 은 안나온 놈!!!
OneVsRestClassifier 은 안나온 놈!!!
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.7374
Perceptron 의 정답률 : 0.6138
QuadraticDiscriminantAnalysis 의 정답률 : 0.8188
RadiusNeighborsClassifier 의 정답률 : 0.8033
RandomForestClassifier 의 정답률 : 0.83
RidgeClassifier 의 정답률 : 0.8062
RidgeClassifierCV 의 정답률 : 0.8062
SGDClassifier 의 정답률 : 0.7992
SVC 의 정답률 : 0.816
StackingClassifier 은 안나온 놈!!!
VotingClassifier 은 안나온 놈!!!  
"""        
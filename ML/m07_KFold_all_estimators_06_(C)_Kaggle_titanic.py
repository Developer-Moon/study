from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
import numpy as np
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)
test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) 
y = np.array(y).reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)
    
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=99)
             
             
#2. 모델구성
all_Algorithms = all_estimators(type_filter='classifier')  # 분류모델
# all_Algorithms = all_estimators(type_filter='regressor') # 회귀모델
print('allAlgorithms: ', all_Algorithms)
print('모델의 개수: ', len(all_Algorithms)) # 41

for (name, algorithm) in all_Algorithms:
    try:
        model = algorithm()
        ypred = cross_val_predict(model, x_test, y_test, cv=kfold)
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        print(name, '의 정답률: ', round(np.mean(score),4))
        
    except:
        # continue # 또는 pass
        print(name, '은 안나온 놈')

# AdaBoostClassifier 의 정답률:  0.8132
# BaggingClassifier 의 정답률:  0.7837
# BernoulliNB 의 정답률:  0.7852
# CalibratedClassifierCV 의 정답률:  0.8048
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.7837
# DecisionTreeClassifier 의 정답률:  0.7627
# DummyClassifier 의 정답률:  0.6305
# ExtraTreeClassifier 의 정답률:  0.7739
# ExtraTreesClassifier 의 정답률:  0.8005
# GaussianNB 의 정답률:  0.7964
# GaussianProcessClassifier 의 정답률:  0.8146
# GradientBoostingClassifier 의 정답률:  0.8146
# HistGradientBoostingClassifier 의 정답률:  0.8314
# KNeighborsClassifier 의 정답률:  0.7935
# LabelPropagation 의 정답률:  0.8188
# LabelSpreading 의 정답률:  0.8188
# LinearDiscriminantAnalysis 의 정답률:  0.809
# LinearSVC 의 정답률:  0.8062
# LogisticRegression 의 정답률:  0.8076
# LogisticRegressionCV 의 정답률:  0.8076
# MLPClassifier 의 정답률:  0.816
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.8076
# NearestCentroid 의 정답률:  0.7978
# NuSVC 의 정답률:  0.816
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.6527
# Perceptron 의 정답률:  0.7654
# QuadraticDiscriminantAnalysis 의 정답률:  0.8103
# RadiusNeighborsClassifier 의 정답률:  0.8034
# RandomForestClassifier 의 정답률:  0.8061
# RidgeClassifier 의 정답률:  0.809
# RidgeClassifierCV 의 정답률:  0.809
# SGDClassifier 의 정답률:  0.7977
# SVC 의 정답률:  0.8188
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈
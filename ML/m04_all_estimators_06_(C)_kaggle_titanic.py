from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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

#2. 모델구성
all_Algorithms = all_estimators(type_filter='classifier')  # 분류모델
# all_Algorithms = all_estimators(type_filter='regressor')  # 회귀모델
# print(all_Algorithms) 전체 모델 보기
print('모델의 갯수 :', len(all_Algorithms)) # 모델의 갯수 :  41

for (name, algorithms) in all_Algorithms:   # (key, value)
    try:                                    # try 예외처리
        model = algorithms()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 :', acc)
    except:
        # continue
        print(name, '은 안나온 놈!!!')

# 버전에 따라 안돌아가는 모델들이 있다

# 모델의 갯수 : 41
# AdaBoostClassifier 의 정답률 : 0.7653631284916201
# BaggingClassifier 의 정답률 : 0.7541899441340782
# BernoulliNB 의 정답률 : 0.7430167597765364
# CalibratedClassifierCV 의 정답률 : 0.7486033519553073
# CategoricalNB 은 안나온 놈!!!
# ClassifierChain 은 안나온 놈!!!
# ComplementNB 의 정답률 : 0.6368715083798883
# DecisionTreeClassifier 의 정답률 : 0.7430167597765364
# DummyClassifier 의 정답률 : 0.5586592178770949
# ExtraTreeClassifier 의 정답률 : 0.7486033519553073
# ExtraTreesClassifier 의 정답률 : 0.7486033519553073
# GaussianNB 의 정답률 : 0.7541899441340782
# GaussianProcessClassifier 의 정답률 : 0.6759776536312849
# GradientBoostingClassifier 의 정답률 : 0.776536312849162
# HistGradientBoostingClassifier 의 정답률 : 0.7821229050279329
# KNeighborsClassifier 의 정답률 : 0.6759776536312849
# LabelPropagation 의 정답률 : 0.6536312849162011
# LabelSpreading 의 정답률 : 0.6536312849162011
# LinearDiscriminantAnalysis 의 정답률 : 0.770949720670391
# LinearSVC 의 정답률 : 0.659217877094972
# LogisticRegression 의 정답률 : 0.770949720670391
# LogisticRegressionCV 의 정답률 : 0.770949720670391
# MLPClassifier 의 정답률 : 0.7821229050279329
# MultiOutputClassifier 은 안나온 놈!!!
# MultinomialNB 의 정답률 : 0.6312849162011173
# NearestCentroid 의 정답률 : 0.6145251396648045
# NuSVC 의 정답률 : 0.7653631284916201
# OneVsOneClassifier 은 안나온 놈!!!
# OneVsRestClassifier 은 안나온 놈!!!
# OutputCodeClassifier 은 안나온 놈!!!
# PassiveAggressiveClassifier 의 정답률 : 0.5865921787709497
# Perceptron 의 정답률 : 0.6815642458100558
# QuadraticDiscriminantAnalysis 의 정답률 : 0.7653631284916201
# RadiusNeighborsClassifier 은 안나온 놈!!!
# RandomForestClassifier 의 정답률 : 0.770949720670391
# RidgeClassifier 의 정답률 : 0.770949720670391
# RidgeClassifierCV 의 정답률 : 0.770949720670391
# SGDClassifier 의 정답률 : 0.6480446927374302
# SVC 의 정답률 : 0.6312849162011173
# StackingClassifier 은 안나온 놈!!!
# VotingClassifier 은 안나온 놈!!!
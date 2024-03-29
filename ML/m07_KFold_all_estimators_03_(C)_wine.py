from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_wine
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
         
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

# AdaBoostClassifier 의 정답률:  0.8941
# BaggingClassifier 의 정답률:  0.93
# BernoulliNB 의 정답률:  0.3798
# CalibratedClassifierCV 의 정답률:  0.9793
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.8803
# DecisionTreeClassifier 의 정답률:  0.8877
# DummyClassifier 의 정답률:  0.4222
# ExtraTreeClassifier 의 정답률:  0.8305
# ExtraTreesClassifier 의 정답률:  0.9862
# GaussianNB 의 정답률:  0.9722
# GaussianProcessClassifier 의 정답률:  0.9722
# GradientBoostingClassifier 의 정답률:  0.93
# HistGradientBoostingClassifier 의 정답률:  0.965
# KNeighborsClassifier 의 정답률:  0.9441
# LabelPropagation 의 정답률:  0.9436
# LabelSpreading 의 정답률:  0.9436
# LinearDiscriminantAnalysis 의 정답률:  0.986
# LinearSVC 의 정답률:  0.9793
# LogisticRegression 의 정답률:  0.9793
# LogisticRegressionCV 의 정답률:  0.9862
# MLPClassifier 의 정답률:  0.9653
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.8951
# NearestCentroid 의 정답률:  0.9579
# NuSVC 의 정답률:  0.9722
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.9862
# Perceptron 의 정답률:  0.9791
# QuadraticDiscriminantAnalysis 의 정답률:  0.9857
# RadiusNeighborsClassifier 은 안나온 놈
# RandomForestClassifier 의 정답률:  0.9722
# RidgeClassifier 의 정답률:  0.9862
# RidgeClassifierCV 의 정답률:  0.9862
# SGDClassifier 의 정답률:  0.9722
# SVC 의 정답률:  0.9931
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈
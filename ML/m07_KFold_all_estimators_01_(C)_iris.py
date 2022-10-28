from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_iris
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = load_iris()
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

# AdaBoostClassifier 의 정답률:  0.9083
# BaggingClassifier 의 정답률:  0.9333
# BernoulliNB 의 정답률:  0.3333
# CalibratedClassifierCV 의 정답률:  0.8583
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 은 안나온 놈
# DecisionTreeClassifier 의 정답률:  0.9083
# DummyClassifier 의 정답률:  0.3083
# ExtraTreeClassifier 의 정답률:  0.9167
# ExtraTreesClassifier 의 정답률:  0.9333
# GaussianNB 의 정답률:  0.9583
# GaussianProcessClassifier 의 정답률:  0.9083
# GradientBoostingClassifier 의 정답률:  0.9333
# HistGradientBoostingClassifier 의 정답률:  0.9167
# KNeighborsClassifier 의 정답률:  0.95
# LabelPropagation 의 정답률:  0.9333
# LabelSpreading 의 정답률:  0.9333
# LinearDiscriminantAnalysis 의 정답률:  0.975
# LinearSVC 의 정답률:  0.9167
# LogisticRegression 의 정답률:  0.9
# LogisticRegressionCV 의 정답률:  0.9583
# MLPClassifier 의 정답률:  0.9
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 은 안나온 놈
# NearestCentroid 의 정답률:  0.925
# NuSVC 의 정답률:  0.95
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.8667
# Perceptron 의 정답률:  0.775
# QuadraticDiscriminantAnalysis 의 정답률:  0.975
# RadiusNeighborsClassifier 의 정답률:  0.475
# RandomForestClassifier 의 정답률:  0.925
# RidgeClassifier 의 정답률:  0.8083
# RidgeClassifierCV 의 정답률:  0.7833
# SGDClassifier 의 정답률:  0.9083
# SVC 의 정답률:  0.9583
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_breast_cancer
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = load_breast_cancer()
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

# AdaBoostClassifier 의 정답률:  0.9648
# BaggingClassifier 의 정답률:  0.9429
# BernoulliNB 의 정답률:  0.611
# CalibratedClassifierCV 의 정답률:  0.9758
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 은 안나온 놈
# DecisionTreeClassifier 의 정답률:  0.9297
# DummyClassifier 의 정답률:  0.622
# ExtraTreeClassifier 의 정답률:  0.9187
# ExtraTreesClassifier 의 정답률:  0.9692
# GaussianNB 의 정답률:  0.9319
# GaussianProcessClassifier 의 정답률:  0.9582
# GradientBoostingClassifier 의 정답률:  0.9516
# HistGradientBoostingClassifier 의 정답률:  0.9604
# KNeighborsClassifier 의 정답률:  0.9692
# LabelPropagation 의 정답률:  0.9736
# LabelSpreading 의 정답률:  0.9714
# LinearDiscriminantAnalysis 의 정답률:  0.956
# LinearSVC 의 정답률:  0.9758
# LogisticRegression 의 정답률:  0.967
# LogisticRegressionCV 의 정답률:  0.9758
# MLPClassifier 의 정답률:  0.9758
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 은 안나온 놈
# NearestCentroid 의 정답률:  0.9407
# NuSVC 의 정답률:  0.9473
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.9714
# Perceptron 의 정답률:  0.9604
# QuadraticDiscriminantAnalysis 의 정답률:  0.956
# RadiusNeighborsClassifier 의 정답률:  nan
# RandomForestClassifier 의 정답률:  0.9626
# RidgeClassifier 의 정답률:  0.9604
# RidgeClassifierCV 의 정답률:  0.9604
# SGDClassifier 의 정답률:  0.9714
# SVC 의 정답률:  0.9802
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈
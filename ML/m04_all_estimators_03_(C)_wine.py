from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)
         
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
             
             
#2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')  # 분류모델 
# allAlgorithms = all_estimators(type_filter='regressor') # 회귀모델 
print('allAlgorithms: ', allAlgorithms)   # 전체 모델 보기
print('모델의 개수: ', len(allAlgorithms)) # 41

for (name, algorithm) in allAlgorithms:   # (key, value)
    try:                                  # try 예외처리
        model = algorithm()
        model.fit(x_train, y_train)
        ypred = model.predict(x_test)
        acc = accuracy_score(y_test, ypred)
        print(name, '의 정답률: ', acc)
    except:
        # continue # 또는 pass
        print(name, '은 안나온 놈')

# 버전에 따라 안돌아가는 모델들이 있다

# 모델의 개수:  41
# AdaBoostClassifier 의 정답률:  0.9166666666666666
# BaggingClassifier 의 정답률:  1.0
# BernoulliNB 의 정답률:  0.3055555555555556
# CalibratedClassifierCV 의 정답률:  1.0
# CategoricalNB 의 정답률:  0.3333333333333333
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.9444444444444444
# DecisionTreeClassifier 의 정답률:  0.9722222222222222
# DummyClassifier 의 정답률:  0.3055555555555556
# ExtraTreeClassifier 의 정답률:  0.8888888888888888
# ExtraTreesClassifier 의 정답률:  1.0
# GaussianNB 의 정답률:  1.0
# GaussianProcessClassifier 의 정답률:  1.0
# GradientBoostingClassifier 의 정답률:  0.9722222222222222
# HistGradientBoostingClassifier 의 정답률:  1.0
# KNeighborsClassifier 의 정답률:  1.0
# LabelPropagation 의 정답률:  1.0
# LabelSpreading 의 정답률:  1.0
# LinearDiscriminantAnalysis 의 정답률:  1.0
# LinearSVC 의 정답률:  1.0
# LogisticRegression 의 정답률:  0.9722222222222222
# LogisticRegressionCV 의 정답률:  0.9722222222222222
# MLPClassifier 의 정답률:  1.0
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.8888888888888888
# NearestCentroid 의 정답률:  1.0
# NuSVC 의 정답률:  1.0
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  1.0
# Perceptron 의 정답률:  1.0
# QuadraticDiscriminantAnalysis 의 정답률:  1.0
# RadiusNeighborsClassifier 의 정답률:  0.8888888888888888
# RandomForestClassifier 의 정답률:  1.0
# RidgeClassifier 의 정답률:  1.0
# RidgeClassifierCV 의 정답률:  1.0
# SGDClassifier 의 정답률:  1.0
# SVC 의 정답률:  1.0
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈
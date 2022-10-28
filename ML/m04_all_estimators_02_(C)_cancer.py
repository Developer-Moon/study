from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
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
# AdaBoostClassifier 의 정답률:  0.9649122807017544
# BaggingClassifier 의 정답률:  0.9473684210526315
# BernoulliNB 의 정답률:  0.631578947368421
# CalibratedClassifierCV 의 정답률:  0.9649122807017544
# CategoricalNB 의 정답률:  0.6578947368421053
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.8771929824561403
# DecisionTreeClassifier 의 정답률:  0.9385964912280702
# DummyClassifier 의 정답률:  0.6491228070175439
# ExtraTreeClassifier 의 정답률:  0.9035087719298246
# ExtraTreesClassifier 의 정답률:  0.9736842105263158
# GaussianNB 의 정답률:  0.9385964912280702
# GaussianProcessClassifier 의 정답률:  0.9649122807017544
# GradientBoostingClassifier 의 정답률:  0.956140350877193
# HistGradientBoostingClassifier 의 정답률:  0.9649122807017544
# KNeighborsClassifier 의 정답률:  0.9649122807017544
# LabelPropagation 의 정답률:  0.9649122807017544
# LabelSpreading 의 정답률:  0.9649122807017544
# LinearDiscriminantAnalysis 의 정답률:  0.9649122807017544
# LinearSVC 의 정답률:  0.9736842105263158
# LogisticRegression 의 정답률:  0.9649122807017544
# LogisticRegressionCV 의 정답률:  0.9736842105263158
# MLPClassifier 의 정답률:  0.9824561403508771
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.8421052631578947
# NearestCentroid 의 정답률:  0.9385964912280702
# NuSVC 의 정답률:  0.956140350877193
# OneVsOneClassifier 은 안나온 놈
# OneVsRestClassifier 은 안나온 놈
# OutputCodeClassifier 은 안나온 놈
# PassiveAggressiveClassifier 의 정답률:  0.9649122807017544
# Perceptron 의 정답률:  0.9824561403508771
# QuadraticDiscriminantAnalysis 의 정답률:  0.9473684210526315
# RadiusNeighborsClassifier 의 정답률:  0.9122807017543859
# RandomForestClassifier 의 정답률:  0.9736842105263158
# RidgeClassifier 의 정답률:  0.956140350877193
# RidgeClassifierCV 의 정답률:  0.9649122807017544
# SGDClassifier 의 정답률:  0.9736842105263158
# SVC 의 정답률:  0.9736842105263158
# StackingClassifier 은 안나온 놈
# VotingClassifier 은 안나온 놈
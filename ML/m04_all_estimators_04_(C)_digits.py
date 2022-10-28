from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_digits
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = load_digits()
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
# AdaBoostClassifier 의 정답률:  0.2972222222222222
# BaggingClassifier 의 정답률:  0.9472222222222222
# BernoulliNB 의 정답률:  0.8333333333333334
# CalibratedClassifierCV 의 정답률:  0.9638888888888889
# CategoricalNB 은 안나온 놈
# ClassifierChain 은 안나온 놈
# ComplementNB 의 정답률:  0.7972222222222223
# DecisionTreeClassifier 의 정답률:  0.8444444444444444
# DummyClassifier 의 정답률:  0.08055555555555556
# ExtraTreeClassifier 의 정답률:  0.7916666666666666
# ExtraTreesClassifier 의 정답률:  0.9833333333333333
# GaussianNB 의 정답률:  0.8277777777777777
# GaussianProcessClassifier 의 정답률:  0.9833333333333333
# GradientBoostingClassifier 의 정답률:  0.9666666666666667
# HistGradientBoostingClassifier 의 정답률:  0.9777777777777777
# KNeighborsClassifier 의 정답률:  0.9888888888888889
# LabelPropagation 의 정답률:  0.9888888888888889
# LabelSpreading 의 정답률:  0.9888888888888889
# LinearDiscriminantAnalysis 의 정답률:  0.9527777777777777
# LogisticRegression 의 정답률:  0.9583333333333334
# LogisticRegressionCV 의 정답률:  0.9666666666666667
# MLPClassifier 의 정답률:  0.9666666666666667
# MultiOutputClassifier 은 안나온 놈
# MultinomialNB 의 정답률:  0.8805555555555555
# NearestCentroid 의 정답률:  0.8888888888888888
# NuSVC 의 정답률:  0.9555555555555556
# OneVsOneClassifier 은 안나온 놈     
# OneVsRestClassifier 은 안나온 놈    
# OutputCodeClassifier 은 안나온 놈   
# PassiveAggressiveClassifier 의 정답률:  0.9555555555555556
# Perceptron 의 정답률:  0.9527777777777777
# QuadraticDiscriminantAnalysis 의 정답률:  0.8666666666666667
# RadiusNeighborsClassifier 은 안나온 놈
# RandomForestClassifier 의 정답률:  0.975
# RidgeClassifier 의 정답률:  0.95        
# RidgeClassifierCV 의 정답률:  0.95
# SGDClassifier 의 정답률:  0.9555555555555556
# SVC 의 정답률:  0.9888888888888889
# StackingClassifier 은 안나온 놈   
# VotingClassifier 은 안나온 놈 
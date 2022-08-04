from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from sklearn.metrics import r2_score, accuracy_score
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
import sklearn as sk
print(sk.__version__)             # 0.24.2
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test =  train_test_split(x, y, train_size=0.9, shuffle=True, random_state=86)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
all_Algorithms = all_estimators(type_filter='classifier') # 분류모델
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
        

"""
모델의 갯수 : 41
AdaBoostClassifier 의 정답률 : 0.2388888888888889
BaggingClassifier 의 정답률 : 0.9333333333333333
BernoulliNB 의 정답률 : 0.8611111111111112
CalibratedClassifierCV 의 정답률 : 0.9666666666666667
CategoricalNB 은 안나온 놈!!!
ClassifierChain 은 안나온 놈!!!
ComplementNB 의 정답률 : 0.8277777777777777
DecisionTreeClassifier 의 정답률 : 0.8555555555555555
DummyClassifier 의 정답률 : 0.08333333333333333
ExtraTreeClassifier 의 정답률 : 0.8
ExtraTreesClassifier 의 정답률 : 0.9833333333333333
GaussianNB 의 정답률 : 0.8611111111111112
GaussianProcessClassifier 의 정답률 : 0.9944444444444445
GradientBoostingClassifier 의 정답률 : 0.9611111111111111
HistGradientBoostingClassifier 의 정답률 : 0.9888888888888889
KNeighborsClassifier 의 정답률 : 0.9944444444444445
LabelPropagation 의 정답률 : 0.9888888888888889
LabelSpreading 의 정답률 : 0.9888888888888889
LinearDiscriminantAnalysis 의 정답률 : 0.9722222222222222
LinearSVC 의 정답률 : 0.9666666666666667
LogisticRegression 의 정답률 : 0.9444444444444444
LogisticRegressionCV 의 정답률 : 0.9555555555555556
MLPClassifier 의 정답률 : 0.9722222222222222  
MultiOutputClassifier 은 안나온 놈!!!
MultinomialNB 의 정답률 : 0.9
NearestCentroid 의 정답률 : 0.9055555555555556
NuSVC 의 정답률 : 0.9611111111111111


OneVsOneClassifier 은 안나온 놈!!!  
OneVsRestClassifier 은 안나온 놈!!! 
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.95
Perceptron 의 정답률 : 0.9555555555555556
QuadraticDiscriminantAnalysis 의 정답률 : 0.8555555555555555
RadiusNeighborsClassifier 은 안나온 놈!!!
RandomForestClassifier 의 정답률 : 0.9833333333333333
RidgeClassifier 의 정답률 : 0.95
RidgeClassifierCV 의 정답률 : 0.95
SGDClassifier 의 정답률 : 0.9611111111111111
SVC 의 정답률 : 0.9888888888888889
StackingClassifier 은 안나온 놈!!!
VotingClassifier 은 안나온 놈!!!  
"""        
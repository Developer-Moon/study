from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from sklearn.metrics import r2_score, accuracy_score
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
import sklearn as sk
print(sk.__version__)             # 0.24.2
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target 
print(x.shape) # (178, 13)
print(y.shape) # (178,)
# print(np.unique(x))


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
AdaBoostClassifier 의 정답률 : 0.8888888888888888
BaggingClassifier 의 정답률 : 1.0
BernoulliNB 의 정답률 : 0.4444444444444444
CalibratedClassifierCV 의 정답률 : 1.0
CategoricalNB 의 정답률 : 0.4444444444444444
ClassifierChain 은 안나온 놈!!!
ComplementNB 의 정답률 : 0.8888888888888888
DecisionTreeClassifier 의 정답률 : 0.8888888888888888
DummyClassifier 의 정답률 : 0.4444444444444444
ExtraTreeClassifier 의 정답률 : 0.8333333333333334
ExtraTreesClassifier 의 정답률 : 1.0
GaussianNB 의 정답률 : 1.0
GaussianProcessClassifier 의 정답률 : 1.0
GradientBoostingClassifier 의 정답률 : 0.8888888888888888
HistGradientBoostingClassifier 의 정답률 : 0.8888888888888888
KNeighborsClassifier 의 정답률 : 1.0
LabelPropagation 의 정답률 : 1.0
LabelSpreading 의 정답률 : 1.0
LinearDiscriminantAnalysis 의 정답률 : 1.0
LinearSVC 의 정답률 : 1.0
LogisticRegression 의 정답률 : 1.0
LogisticRegressionCV 의 정답률 : 1.0
MLPClassifier 의 정답률 : 1.0
MultiOutputClassifier 은 안나온 놈!!!
MultinomialNB 의 정답률 : 1.0
NearestCentroid 의 정답률 : 0.9444444444444444
NuSVC 의 정답률 : 1.0
OneVsOneClassifier 은 안나온 놈!!!
OneVsRestClassifier 은 안나온 놈!!!
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 : 1.0
Perceptron 의 정답률 : 1.0
QuadraticDiscriminantAnalysis 의 정답률 : 0.9444444444444444
RadiusNeighborsClassifier 의 정답률 : 0.9444444444444444
RandomForestClassifier 의 정답률 : 1.0
RidgeClassifier 의 정답률 : 1.0
RidgeClassifierCV 의 정답률 : 1.0
SGDClassifier 의 정답률 : 1.0
SVC 의 정답률 : 1.0


StackingClassifier 은 안나온 놈!!!
VotingClassifier 은 안나온 놈!!!
"""        
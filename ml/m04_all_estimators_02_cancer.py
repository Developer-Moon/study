from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
all_Algorithms = all_estimators(type_filter='classifier')   # 분류모델 
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
모델의 갯수 :  41
AdaBoostClassifier 의 정답률 : 0.9473684210526315
BaggingClassifier 의 정답률 : 0.9385964912280702
BernoulliNB 의 정답률 : 0.6403508771929824
CalibratedClassifierCV 의 정답률 : 0.9649122807017544
CategoricalNB 은 안나온 놈!!!
ClassifierChain 은 안나온 놈!!!
ComplementNB 의 정답률 : 0.7807017543859649
DecisionTreeClassifier 의 정답률 : 0.9385964912280702
DummyClassifier 의 정답률 : 0.6403508771929824
ExtraTreeClassifier 의 정답률 : 0.9385964912280702
ExtraTreesClassifier 의 정답률 : 0.9649122807017544
GaussianNB 의 정답률 : 0.9210526315789473
GaussianProcessClassifier 의 정답률 : 0.9649122807017544
GradientBoostingClassifier 의 정답률 : 0.956140350877193
HistGradientBoostingClassifier 의 정답률 : 0.9736842105263158
KNeighborsClassifier 의 정답률 : 0.956140350877193
LabelPropagation 의 정답률 : 0.9473684210526315
LabelSpreading 의 정답률 : 0.9473684210526315
LinearDiscriminantAnalysis 의 정답률 : 0.9473684210526315
LinearSVC 의 정답률 : 0.9736842105263158
LogisticRegression 의 정답률 : 0.9649122807017544
LogisticRegressionCV 의 정답률 : 0.9736842105263158
MLPClassifier 의 정답률 : 0.9649122807017544
MultiOutputClassifier 은 안나온 놈!!!
MultinomialNB 의 정답률 : 0.8508771929824561
NearestCentroid 의 정답률 : 0.9298245614035088
NuSVC 의 정답률 : 0.9473684210526315
OneVsOneClassifier 은 안나온 놈!!!
OneVsRestClassifier 은 안나온 놈!!!
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.9298245614035088
Perceptron 의 정답률 : 0.9736842105263158
QuadraticDiscriminantAnalysis 의 정답률 : 0.9385964912280702
RadiusNeighborsClassifier 은 안나온 놈!!!
RandomForestClassifier 의 정답률 : 0.9649122807017544
RidgeClassifier 의 정답률 : 0.9473684210526315
RidgeClassifierCV 의 정답률 : 0.9473684210526315
SGDClassifier 의 정답률 : 0.9912280701754386
SVC 의 정답률 : 0.9736842105263158


StackingClassifier 은 안나온 놈!!!
VotingClassifier 은 안나온 놈!!!
"""        
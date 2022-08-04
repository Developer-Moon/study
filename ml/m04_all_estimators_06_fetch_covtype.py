from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import r2_score, accuracy_score
#--------------------------------------------------------------------------------#
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore') # warning 출력X
#--------------------------------------------------------------------------------#


# 1. 데이터
datasets = fetch_covtype()
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
AdaBoostClassifier 의 정답률 : 0.560686379126364
BaggingClassifier 의 정답률 : 0.963340332518674
BernoulliNB 의 정답률 : 0.6289628584213969
CalibratedClassifierCV 의 정답률 : 0.709528071322846
CategoricalNB 의 정답률 : 0.6303741695638705
ClassifierChain 은 안나온 놈!!!
ComplementNB 의 정답률 : 0.6176723692816082
DecisionTreeClassifier 의 정답률 : 0.9424804653884548
DummyClassifier 의 정답률 : 0.49263364428074763
ExtraTreeClassifier 의 정답률 : 0.8760972083577158
ExtraTreesClassifier 의 정답률 : 0.9543217100960381
GaussianNB 의 정답률 : 0.08929124642869436
GaussianProcessClassifier 은 안나온 놈!!!
GradientBoostingClassifier 의 정답률 : 0.7693194726515439
HistGradientBoostingClassifier 의 정답률 : 0.8250834738907439
KNeighborsClassifier 의 정답률 : 0.9376441430587588
LabelPropagation 은 안나온 놈!!!
LabelSpreading 은 안나온 놈!!!
LinearDiscriminantAnalysis 의 정답률 : 0.6778596261746583
LinearSVC 의 정답률 : 0.7091322157584937
LogisticRegression 의 정답률 : 0.7173074937179443
LogisticRegressionCV 의 정답률 : 0.7222298716051082
MLPClassifier 의 정답률 : 0.8392826408729476
MultiOutputClassifier 은 안나온 놈!!!       
MultinomialNB 의 정답률 : 0.6398402808853396
NearestCentroid 의 정답률 : 0.3861657085814602
NuSVC 은 안나온 놈!!!
OneVsOneClassifier 은 안나온 놈!!!  
OneVsRestClassifier 은 안나온 놈!!! 
OutputCodeClassifier 은 안나온 놈!!!
PassiveAggressiveClassifier 의 정답률 : 0.5645760903239131
Perceptron 의 정답률 : 0.6065884134797426
QuadraticDiscriminantAnalysis 의 정답률 : 0.10600323568896079
RadiusNeighborsClassifier 의 정답률 : 0.6494440810987574
RandomForestClassifier 의 정답률 : 0.9557674434614988
RidgeClassifier 의 정답률 : 0.6976868266152628
RidgeClassifierCV 의 정답률 : 0.6976524043922756
SGDClassifier 의 정답률 : 0.7076692712815393
"""        
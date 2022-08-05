from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, StratifiedKFold, GridSearchCV # 격자탐색, Cross_Validation
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # 논리회귀(분류)
from sklearn.neighbors import KNeighborsClassifier              # 최근접 이웃
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier             # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
        
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3, 4, 5]},                             # 12번 key, value 형태
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},                                 #  6번
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]} # 24번 - 12+6+24 : 총42번 훈련
]
                      
#2. 모델구성
model = SVC(C=10, kernel='linear', degree=3)
#model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) # 42번 X 5(n_splits=5) = 210, n_jobs = cpu 개수 몇개 사용 -1은 전부 다,  4는 4개
# refit=True면 최적의 파라미터로 훈련, False면 해당 파라미터로 훈련하지 않고 마지막 파라미터로 훈련

#3. 컴파일, 훈련
model.fit(x_train, y_train)
# print("최적의 매개변수 :", model.best_estimator_) # best_estimator_ 가장 좋은 추정지
     # 최적의 매개변수 : SVC(C=10, kernel='linear')

# # print("최적의 파라미터 :", model.best_params_)
     # 최적의 파라미터 : {'C': 10, 'degree': 3, 'kernel': 'linear'}
     
#  print("best_score :", model.best_score_)
     # best_score : 0.9666666666666668  model.fit(x_train, y_train) test가 없는 값

print('model.score :', model.score(x_test, y_test)) 
    # model.score : 1.0

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))  
     # accuracy_score : 1.0                                               
     
# y_predict_best = model.best_estimator_.predict(x_test)       
# print("최적 튠 ACC :", accuracy_score(y_test, y_predict_best)) 
     # 최적 튠 ACC : 1.0
     
# score, acc_score, best_estimator_ 모두 같다, 확실하게 하기 위해 best_estimator_ 로 확인
     
# 5개의 folds 42개의 파라미터 후보  총 210번 의 훈련을 한다
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# acc : 1.0   
     
     
# model.score : 1.0
# accuracy_score : 1.0
     
     
     
     
     
#4. 평가, 예측
# score = model.score(x_test, y_test)
# print('acc :', score)

# LinearSVC - acc : 1.0
# SVC - acc : 1.0
# Perceptron - acc : 0.9666666666666667
# LogisticRegression - acc : 1.0
# KNeighborsClassifie r- acc : 1.0
# DecisionTreeClassifier - acc : 1.0
# RandomForestClassifier - acc : 1.0




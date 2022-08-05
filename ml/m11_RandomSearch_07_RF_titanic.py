from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV # 격자탐색, Cross_Validation
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
import time
#----------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import RandomForestClassifier
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')


train_set = train_set.drop(columns='Cabin', axis=1)
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True)
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True)
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

# train_set 불러올 때와 마찬가지로 전처리시켜야 model.predict에 넣어서 y값 구하기가 가능함-----------
test_set = test_set.drop(columns='Cabin', axis=1)
test_set['Age'].fillna(test_set['Age'].mean(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].mean(), inplace=True)

test_set['Embarked'].fillna(test_set['Embarked'].mode()[0], inplace=True)
test_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
test_set = test_set.drop(columns = ['PassengerId','Name','Ticket'],axis=1)
#---------------------------------------------------------------------------------------------------

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1) 
y = np.array(y).reshape(-1, 1) # 벡터로 표시되어 있는 y데이터를 행렬로 전환

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
        
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12, 14, 16]},       # 2 x 6 = 12번                                       
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10, 13]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10, 15, 20]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4, 8]},
    {'n_jobs' : [-1, 2, 4], 'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10], 'n_estimators' : [100, 200, 300]}
]


"""
parameters = [
    {'n_estimators' : [100, 200], },  # 에포
    {'max_depth' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [-1, 2, 4]}
]
"""   
                  
#2. 모델구성
# model = SVC(C=10, kernel='linear', degree=3)
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) # 42번 X 5(n_splits=5) = 210, n_jobs = cpu 개수 몇개 사용 -1은 전부 다,  4는 4개
# refit=True면 최적의 파라미터로 훈련, False면 해당 파라미터로 훈련하지 않고 마지막 파라미터로 훈련

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()
print("최적의 매개변수 :", model.best_estimator_) # best_estimator_ 가장 좋은 추정지
     # 최적의 매개변수 : SVC(C=10, kernel='linear')

print("최적의 파라미터 :", model.best_params_)
     # 최적의 파라미터 : {'C': 10, 'degree': 3, 'kernel': 'linear'}
     
print("best_score :", model.best_score_)
     # best_score : 0.9666666666666668  model.fit(x_train, y_train) test가 없는 값

print('model.score :', model.score(x_test, y_test)) 
    # model.score : 1.0

y_predict = model.predict(x_test)
print('accuracy_score :', accuracy_score(y_test, y_predict))  
     # accuracy_score : 1.0                                               
     
y_predict_best = model.best_estimator_.predict(x_test)       
print("최적 튠 ACC :", accuracy_score(y_test, y_predict_best)) 
     # 최적 튠 ACC : 1.0
 
print('걸린시간 :', end - start)     
# score, acc_score, best_estimator_ 모두 같다, 확실하게 하기 위해 best_estimator_ 로 확인
     
# 5개의 folds 42개의 파라미터 후보  총 210번 의 훈련을 한다
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# acc : 1.0   
     
     
# 최적의 매개변수 : RandomForestClassifier(min_samples_leaf=3, n_jobs=2)
# 최적의 파라미터 : {'n_jobs': 2, 'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 3}
# best_score : 0.8357037328868315
# model.score : 0.7597765363128491
# accuracy_score : 0.7597765363128491
# 최적 튠 ACC : 0.7597765363128491
# 걸린시간 : 4.296833753585815

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV # 격자탐색, Cross_Validation
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import time
#----------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import RandomForestRegressor
#----------------------------------------------------------------------------------------------------------------#


## 1. 데이터
path = './_data/ddarung/'                                        
train_set = pd.read_csv(path + 'train.csv', index_col=0)                                                               
test_set = pd.read_csv(path + 'test.csv', index_col=0)     
   
train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   
train_set = train_set.dropna()                

x = train_set.drop(['count'], axis=1)       
y = train_set['count']                      

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=16)
        
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

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
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) # 42번 X 5(n_splits=5) = 210, n_jobs = cpu 개수 몇개 사용 -1은 전부 다,  4는 4개
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
print('accuracy_score :', r2_score(y_test, y_predict))  
     # accuracy_score : 1.0                                               
     
y_predict_best = model.best_estimator_.predict(x_test)       
print("최적 튠 ACC :", r2_score(y_test, y_predict_best)) 
     # 최적 튠 ACC : 1.0
 
print('걸린시간 :', end - start)     
# score, acc_score, best_estimator_ 모두 같다, 확실하게 하기 위해 best_estimator_ 로 확인
     
# 5개의 folds 42개의 파라미터 후보  총 210번 의 훈련을 한다
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# acc : 1.0   
     

# Fitting 5 folds for each of 216 candidates, totalling 1080 fits
# 최적의 매개변수 : RandomForestClassifier(min_samples_leaf=3, min_samples_split=20)
# 최적의 파라미터 : {'min_samples_leaf': 3, 'min_samples_split': 20}
# best_score : 0.9583333333333334
# model.score : 1.0
# accuracy_score : 1.0
# 최적 튠 ACC : 1.0
# 걸린시간 : 34.58961462974548

# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# 최적의 매개변수 : RandomForestRegressor(max_depth=14)
# 최적의 파라미터 : {'n_estimators': 100, 'max_depth': 14}
# best_score : 0.7749457679252236
# model.score : 0.7727740619654554
# accuracy_score : 0.7727740619654554
# 최적 튠 ACC : 0.7727740619654554
# 걸린시간 : 5.742735385894775

     
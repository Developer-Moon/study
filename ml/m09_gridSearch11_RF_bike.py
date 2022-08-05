from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, StratifiedKFold, GridSearchCV # 격자탐색, Cross_Validation
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import fetch_california_housing
import pandas as pd
import time
#----------------------------------------------------------------------------------------------------------------#
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestRegressor           # 결정트리를 여러개 랜덤으로 뽑아서 앙상블해서 본다
#----------------------------------------------------------------------------------------------------------------#


## 1. 데이터
path = './_data/kaggle_bike/'        
train_set = pd.read_csv(path + 'train.csv', index_col=0)   
test_set = pd.read_csv(path + 'test.csv', index_col=0)  

x = train_set.drop(['casual', 'registered', 'count'], axis=1)  
y = train_set['count']   
  
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=5)  
        
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12, 14, 16]},                                             
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
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) # 42번 X 5(n_splits=5) = 210, n_jobs = cpu 개수 몇개 사용 -1은 전부 다,  4는 4개
# refit=True면 최적의 파라미터로 훈련, False면 해당 파라미터로 훈련하지 않고 마지막 파라미터로 훈련

#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()
print("최적의 매개변수 :", model.best_estimator_) # best_estimator_ 가장 좋은 추정지
print("최적의 파라미터 :", model.best_params_)   
print("best_score :", model.best_score_) # test가 없는 값
print('model.score :', model.score(x_test, y_test)) 
y_predict = model.predict(x_test)
print('r2_score :', r2_score(y_test, y_predict))    
y_predict_best = model.best_estimator_.predict(x_test)       
print("최적 튠 r2 :", r2_score(y_test, y_predict_best)) 
print('걸린시간 :', end - start)     

# score, acc_score, best_estimator_ 모두 같다, 확실하게 하기 위해 best_estimator_ 로 확인
     
# Fitting 5 folds for each of 216 candidates, totalling 1080 fits
# 최적의 매개변수 : RandomForestRegressor(max_depth=10, n_estimators=200)
# 최적의 파라미터 : {'max_depth': 10, 'n_estimators': 200}
# best_score : 0.3434500460732054
# model.score : 0.3636512178478768
# r2_score : 0.3636512178478768
# 최적 튠 r2 : 0.3636512178478768
# 걸린시간 : 194.0798900127411
     
     

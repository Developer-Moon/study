from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
import pandas as pd
import time
#----------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
path = './_data/kaggle_bike/'        
train_set = pd.read_csv(path + 'train.csv', index_col=0)   
test_set = pd.read_csv(path + 'test.csv', index_col=0)  

x = train_set.drop(['casual', 'registered', 'count'], axis=1)  
y = train_set['count']   
  
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=5)  
        
n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},                                   
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10, 13]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4, 8]},
    {'n_jobs' : [-1, 2, 4], 'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]}
]

                  
#2. 모델구성
model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) 


#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()
y_predict = model.predict(x_test)
y_predict_best = model.best_estimator_.predict(x_test) 

print("최적의 매개변수 :", model.best_estimator_)
print("최적의 파라미터 :", model.best_params_)
print("best_score :", model.best_score_)
print('model.score :', model.score(x_test, y_test)) 
print('accuracy_score :', r2_score(y_test, y_predict))  
print("최적 튠 ACC :", r2_score(y_test, y_predict_best)) 
print('걸린시간 :', end - start)     


"""
n_iterations: 5
n_required_iterations: 5
n_possible_iterations: 5
min_resources_: 100
max_resources_: 8164
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 108
n_resources: 100
Fitting 5 folds for each of 108 candidates, totalling 540 fits
----------
iter: 1
n_candidates: 36
n_resources: 300
Fitting 5 folds for each of 36 candidates, totalling 180 fits
----------
iter: 2
n_candidates: 12
n_resources: 900
Fitting 5 folds for each of 12 candidates, totalling 60 fits
----------
iter: 3
n_candidates: 4
n_resources: 2700
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 4
n_candidates: 2
n_resources: 8100
Fitting 5 folds for each of 2 candidates, totalling 10 fits
최적의 매개변수 : RandomForestRegressor(min_samples_leaf=10, min_samples_split=5, n_jobs=2)
최적의 파라미터 : {'min_samples_leaf': 10, 'min_samples_split': 5, 'n_jobs': 2}
best_score : 0.3411210603563052
model.score : 0.35172963460512363
accuracy_score : 0.35172963460512363
최적 튠 ACC : 0.35172963460512363
걸린시간 : 21.252169370651245
"""
     
     
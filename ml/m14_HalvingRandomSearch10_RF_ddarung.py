from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import fetch_california_housing
import numpy as np 
import pandas as pd
import time
#----------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터
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
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},                                   
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10, 13]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4, 8]},
    {'n_jobs' : [-1, 2, 4], 'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]}
]

                  
#2. 모델구성
model = HalvingRandomSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) 


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
최적의 매개변수 : RandomForestRegressor(min_samples_leaf=7, min_samples_split=5)
최적의 파라미터 : {'min_samples_leaf': 7, 'min_samples_split': 5}
best_score : 0.3481146280544476
model.score : 0.5557520318623039
accuracy_score : 0.5557520318623039
최적 튠 ACC : 0.5557520318623039
걸린시간 : 17.987809896469116
"""



""" ------------------------------------------------------------------------------ random
최적의 매개변수 : RandomForestRegressor(n_jobs=2)
최적의 파라미터 : {'n_jobs': 2, 'min_samples_split': 2}
best_score : 0.7472233187906554
model.score : 0.7606172368704311
accuracy_score : 0.7606172368704311
최적 튠 ACC : 0.7606172368704311
걸린시간 : 16.724079370498657
"""
     
     
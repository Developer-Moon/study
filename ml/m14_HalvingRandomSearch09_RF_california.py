from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import fetch_california_housing
import time
#----------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = fetch_california_housing()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95,shuffle=True, random_state=9)
        
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
최적의 매개변수 : RandomForestRegressor(max_depth=10)
최적의 파라미터 : {'n_estimators': 100, 'max_depth': 10}
best_score : 0.6876397913196073
model.score : 0.7755750861421876
accuracy_score : 0.7755750861421876
최적 튠 ACC : 0.7755750861421876
걸린시간 : 21.91587471961975
"""
     
     
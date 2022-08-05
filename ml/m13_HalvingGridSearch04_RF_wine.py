from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_wine
import time
#----------------------------------------------------------------------------------------------------------------#
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
#----------------------------------------------------------------------------------------------------------------#


# 1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
        
n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},                                   
    {'max_depth' : [6, 8, 10, 12], 'min_samples_leaf' : [3, 5, 7, 10, 13]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4, 8]},
    {'n_jobs' : [-1, 2, 4], 'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]}
]

                  
#2. 모델구성
model = HalvingGridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1) 


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
print('accuracy_score :', accuracy_score(y_test, y_predict))  
print("최적 튠 ACC :", accuracy_score(y_test, y_predict_best)) 
print('걸린시간 :', end - start)     


"""
n_iterations: 2
n_required_iterations: 5
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 108
n_resources: 30
Fitting 5 folds for each of 108 candidates, totalling 540 fits
----------
iter: 1
n_candidates: 36
n_resources: 90
Fitting 5 folds for each of 36 candidates, totalling 180 fits
최적의 매개변수 : RandomForestClassifier(min_samples_leaf=3, n_jobs=4)
최적의 파라미터 : {'min_samples_leaf': 3, 'min_samples_split': 2, 'n_jobs': 4}
best_score : 0.9882352941176471
model.score : 1.0
accuracy_score : 1.0
최적 튠 ACC : 1.0
걸린시간 : 16.295806169509888
"""
     
     
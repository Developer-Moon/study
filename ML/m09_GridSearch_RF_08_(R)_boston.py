from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time


#1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
'''
랜덤포레스트의 파라미터들!
parameters = [
    {'n_estimators':[100,200]},
    {'max_depth':[6,8,10,12]},
    {'min_samples_leaf':[3,5,7,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4]}
]
'''       


#2. 모델구성
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)


# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

ypred = model.predict(x_test)
ypred_best = model.best_estimator_.predict(x_test)

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)
print('best_score :', model.best_score_)
print('model.score :', model.score(x_test, y_test))
print('r2 score :', r2_score(y_test, ypred))
print('best tuned r2 :', r2_score(y_test, ypred_best))
print('걸린시간 :', round(end-start,2),'초')

# Fitting 5 folds for each of 72 candidates, totalling 360 fits
# 최적의 매개변수 : RandomForestRegressor(max_depth=12)
# 최적의 파라미터 : {'max_depth': 12, 'min_samples_split': 2, 'n_estimators': 100}
# best_score : 0.857145116205434
# model.score : 0.8915234979351654
# r2 score : 0.8915234979351654
# best tuned r2 : 0.8915234979351654
# 걸린시간 : 28.33 초
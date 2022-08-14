from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import time


#1. 데이터
datasets = load_breast_cancer()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

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
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)


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
print('acc score :', accuracy_score(y_test, ypred))
print('best tuned acc :', accuracy_score(y_test, ypred_best))
print('걸린시간 :', round(end-start,2),'초')

# Fitting 5 folds for each of 152 candidates, totalling 760 fits
# 최적의 매개변수 : RandomForestClassifier(max_depth=12, n_jobs=2)
# 최적의 파라미터 : {'max_depth': 12, 'n_estimators': 100, 'n_jobs': 2}
# best_score : 0.956043956043956
# model.score : 0.956140350877193
# acc score : 0.956140350877193
# best tuned acc : 0.956140350877193
# 걸린시간 : 23.21 초
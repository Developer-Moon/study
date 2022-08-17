from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, \
     HalvingGridSearchCV # 현재 버전에서 정식 지원이 안되니까 experimental 임포트
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import time


#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=9)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'C':[1,10,100,1000], 'kernel':['linear'], 'degree':[3,4,5]},                                # 12
    {'C':[1,10,100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},                                 # 6
    {'C':[1,10,100,1000], 'kernel':['sigmoid'], 'gamma':[0.01,0.001,0.0001], 'degree':[3,4]}     # 24
]                                                                                                # 총 42회 파라미터 해봄
                   
                      
#2. 모델구성
model = HalvingGridSearchCV(SVC(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)


# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()


print('최적의 매개변수: ', model.best_estimator_)
print('최적의 파라미터: ', model.best_params_)
print('best_score_: ', model.best_score_)
print('model.score: ', model.score(x_test, y_test))

ypred = model.predict(x_test)
print('acc score: ', accuracy_score(y_test, ypred))

ypred_best = model.best_estimator_.predict(x_test)
print('best tuned acc: ', accuracy_score(y_test, ypred_best))

print('걸린시간: ', round(end-start,2), '초')

# 두번 이상? 놔눠서 훈련한다
# 처음 일부를 적은 자원으로 돌려서 상위권을 뽑아놓고 그것 중에서 한번 더 돌린다

# ----------
# iter: 0
# n_candidates: 42 # 42개의 조합을
# n_resources: 30  # 30개의 리소스로 확인해서 좋은거 뽑고
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14 # 뽑힌 14개를
# n_resources: 90  # 90개의 리소스를 써서 최종값 빼기

# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# 최적의 매개변수:  SVC(C=100, degree=5, kernel='linear')
# 최적의 파라미터:  {'C': 100, 'degree': 5, 'kernel': 'linear'}
# best_score_:  0.9888888888888889
# model.score:  1.0
# acc score:  1.0
# best tuned acc:  1.0
# 걸린시간:  1.95 초
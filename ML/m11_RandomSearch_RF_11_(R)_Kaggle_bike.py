from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, \
    RandomizedSearchCV # 그리드 서치로 파라미터 가져오는 것 중, 랜덤으로 적용해본다
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd
import time


#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True)
test_set.drop('datetime',axis=1,inplace=True)
train_set.drop('casual',axis=1,inplace=True) 
train_set.drop('registered',axis=1,inplace=True) 

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
                                                                                                        
                                    
#2. 모델구성
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, n_iter=15) # n_iter=10 디폴트, 파라미터 조합 중 열가지만 해본다


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

# Fitting 5 folds for each of 15 candidates, totalling 75 fits
# 최적의 매개변수 : RandomForestRegressor(min_samples_leaf=3, min_samples_split=5, n_jobs=-1)
# 최적의 파라미터 : {'n_jobs': -1, 'min_samples_split': 5, 'min_samples_leaf': 3}
# best_score : 0.9439390202758364
# model.score : 0.9470221986777498
# r2 score : 0.9470221986777498
# best tuned r2 : 0.9470221986777498
# 걸린시간 : 14.88 초
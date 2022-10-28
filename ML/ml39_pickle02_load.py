from random import shuffle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score, accuracy_score


#1.데이터 
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target 

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 123)

parameters = {'n_estimators' : [100],
              'learning_rate': [0.1],
              'max_depth': [3],
              'gamma': [1],
              'min_child_weight': [1],
              'subsample': [1],
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              'colsample_bynode': [1] ,
              'reg_alpha': [0],
              'reg_lambda':[1]
              }



#2.모델 
# model = XGBClassifier(random_state=123,
#                       n_estimators=1000,
#                       learning_rate=0.1,
#                       max_depth=3,
#                       gamma=1,
#                     )

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

# model.fit(x_train, y_train, early_stopping_rounds=10, eval_set=[(x_train, y_train), (x_test, y_test)],
#           eval_metric='error', 
#           # 회귀 : rmse, mae, resle
#           # 이진 : error, auc...mlogloss...
#           # 다중이 : merror, mlogloss...
#           ) 

# pickle load
import pickle
path = 'D:/study_data/_save/_xg/'
model = pickle.load(open(path + 'm39_pickle01_save.dat', 'rb'))



results = model.score(x_test, y_test)
print('최종점수 :', results)  # 0.9736842105263158

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('진짜 최종점수 test 점수 :', acc)
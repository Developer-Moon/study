from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris

from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

import warnings
warnings.filterwarnings('ignore')



# LGBMRegressor도 하고
# XGBoost나 cat머시기로 과제하기 








#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델 


# 'n_estimators' : [100, 200, 300, 400, 500, 1000] # 디폴트 100 / 1~inf  (inf: 무한대)
# 'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3/ 0~1 / eta라고 써도 먹힘
# 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~ inf / 정수
# 'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0/ 0~inf
# 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10] 디폴트 1 / 0~inf
# 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0/ 0~inf / L1 절대값 가중치 규제 /alpha
# 'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 /lambda

bayesian_params = {
    'n_estimators' : (100, 1000),   
    'learning_rate' : (0.001, 0.1),   
    
    
}

def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators':500,                              # 무조건 정수형
        'learning_rate':0.02,
        'max_depth':int(round(max_depth)),
        'num_leaves':int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'min_child_weight':int(round(min_child_weight)), # 무조건 정수
        'subsample':max(min(subsample, 1), 0), # 0~1 사이의 값이 들어와야 한다 1이상이면 1
        'colsample_bytree':max(min(colsample_bytree, 1), 0),
        'max_bin':max(int(round(max_bin)), 10), # 10이상의 정수
        'reg_lambda':max(reg_lambda, 0),        # 무조건 양수만
        'reg_alpha':max(reg_alpha, 0)           # 무조건 양수만
    }
    # *여러개의인자를받겠다     
    # **키워드받겠다(딕셔너리형태)
    
    model = XGBClassifier(**params) # 모델을 함수안에 써야한다
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    
    return result

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=1234)

lgb_bo.maximize(init_points=5, n_iter=50)

print(lgb_bo.max)
# {'target': 0.4862677564010508,
#  'params': {'colsample_bytree': 0.5, 'max_bin': 442.51857689138404, 'max_depth': 7.566720006436481, 'min_child_samples': 23.294048382587064,
#             'min_child_weight': 49.396474502081844, 'num_leaves': 24.0, 'reg_alpha': 22.195728846767462, 'reg_lambda': 10.0, 'subsample': 0.5}}
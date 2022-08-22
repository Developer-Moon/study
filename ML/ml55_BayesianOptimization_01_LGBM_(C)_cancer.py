from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------#
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
#------------------------------------------#
# 알고리즘으로 파라미터를 찾는데 - 이 알고리즘을 신뢰할 수 있냐 없냐 이걸 또 나온 파라미터로 돌려서 확인한다


#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델 
bayesian_params = {
    'max_depth' : (8, 10),   
    'num_leaves' : (24, 64),
    'min_child_samples' :(10, 50),
    'min_child_weight' :(2, 10),
    'subsample' : (0.5, 1),
    'colsample_bytree' : (0.5, 0.6),
    'max_bin' : (450, 500),
    'reg_lambda': (0.2, 8),
    'reg_alpha' : (0.01, 20)
}



def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight, subsample, colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators':500,                                 # 무조건 정수형
        'learning_rate':0.02,
        'max_depth':int(round(max_depth)),
        'num_leaves':int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        'min_child_weight':int(round(min_child_weight)),    # 무조건 정수형
        'subsample':max(min(subsample, 1), 0),              # 0~1 사이의 값이 들어와야 한다 1이상이면 1
        'colsample_bytree':max(min(colsample_bytree, 1), 0),
        'max_bin':max(int(round(max_bin)), 10),             # 10이상의 정수
        'reg_lambda':max(reg_lambda, 0),                    # 무조건 양수만
        'reg_alpha':max(reg_alpha, 0)                       # 무조건 양수만
    }
    # *여러개의인자를받겠다     
    # **키워드받겠다(딕셔너리형태)
    
    model = LGBMRegressor(**params) # 모델을 함수안에 써야한다
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    result = r2_score(y_test, y_predict)
    
    return result

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=1234)

lgb_bo.maximize(init_points=5, n_iter=100)

print(lgb_bo.max)


# {'target': 0.6190694981953919,
#  'params': {'colsample_bytree': 0.5580545971517282, 'max_bin': 352.3787406911987, 'max_depth': 8.622177408360672,
#             'min_child_samples': 30.1967820112652, 'min_child_weight': 6.626825838545455, 'num_leaves': 30.428571403804987,
#             'reg_alpha': 3.2579282045866855, 'reg_lambda': 6.687917975673115, 'subsample': 0.9256921192141869}}


# {'target': 0.6244620004074347,
#  'params': {'colsample_bytree': 0.5, 'max_bin': 467.5279426141011, 'max_depth': 10.0, 'min_child_samples': 26.140458826519446,
# 'min_child_weight': 15.370614753704304, 'num_leaves': 44.36553407936451, 'reg_alpha': 10.524827466978067,
# 'reg_lambda': 0.5761097710580022, 'subsample': 0.5}}
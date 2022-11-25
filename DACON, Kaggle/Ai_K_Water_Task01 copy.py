'''
비오면 물 덜 쓴다 - 날씨
여름이라도 장마철에는 물 덜 쓴다

추운날에 관 동파되면 물이 새서 물 많이 나간다

혁신도시면 공장 돌리는 시간에 많이 나간다
설날 추석에 물 많이 씀 - 도심같은 경우는 줄어들고
반도체
밥시간 설거지

기상자료개방포털
'''

import pandas as pd
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt

# Data
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

path = 'D:/study_data/_data/aifactory/Task_01/data_city/'
train_set = pd.read_csv(path + 'data_tr_city.csv')
test_set = pd.read_csv(path + 'data_ts_city.csv')


train_set['datetime'] = pd.to_datetime(train_set['datetime'])
train_set['연도'] = train_set['datetime'].dt.year
train_set['월'] = train_set['datetime'].dt.month
train_set['일'] = train_set['datetime'].dt.day
train_set['시간'] = train_set['datetime'].dt.hour


# train_set['요일'] = 
train_set = train_set.drop(['datetime'], axis=1)




print(tabulate(train_set.head(), headers='keys', numalign='right', stralign='right' ))




train_set = train_set[['연도별', '월별', '일별', '시간별', '구미 혁신도시배수지 유출유량 적산차']]

train_set['구미 혁신도시배수지 유출유량 적산차'] = train_set['구미 혁신도시배수지 유출유량 적산차'].fillna(train_set['구미 혁신도시배수지 유출유량 적산차'].mean())
test_set['구미 혁신도시배수지 유출유량 적산차'] = test_set['구미 혁신도시배수지 유출유량 적산차'].fillna(test_set['구미 혁신도시배수지 유출유량 적산차'].mean())


# 기존값 : -6093821 -> 이상치 - 2018-12-31 15:00 : 577, 2019-01-07 15:00 : 304 평균으로 바꿈
train_set['구미 혁신도시배수지 유출유량 적산차'][17558] = 440 

# 기존값 : -2584551 -> 이상치 - 2017-03-23 16:00 : 101, 2017-04-06 16:00 : 215 평균으로 바꿈
train_set['구미 혁신도시배수지 유출유량 적산차'][2127] = 158



print()


# print(train_set['구미 혁신도시배수지 유출유량 적산차'][2127])
          
#         23         101
# 2017-03-30 16:00	-2584551
#                    215
          

plt.plot(train_set['연도별'], train_set['구미 혁신도시배수지 유출유량 적산차'])
plt.show()









test_set['datetime'] = pd.to_datetime(test_set['datetime'])
test_set['연도별'] = test_set['datetime'].dt.year
test_set['월별'] = test_set['datetime'].dt.month
test_set['일별'] = test_set['datetime'].dt.day
test_set['시간별'] = test_set['datetime'].dt.hour
test_set = test_set.drop(['datetime'], axis=1)
test_set = train_set[['연도별', '월별', '일별', '시간별', '구미 혁신도시배수지 유출유량 적산차']]


# print(tabulate(train_set.head(), headers='keys', numalign='right', stralign='right' ))

# print(train_set.isnull().sum()) 구미 혁신도시배수지 유출유량 적산차    8

# list = train_set[train_set['구미 혁신도시배수지 유출유량 적산차'].isnull()]
# print(list)


print(train_set.isnull().sum()) # 구미 혁신도시배수지 유출유량 적산차    8
print(test_set.isnull().sum())

train_set = np.array(train_set)
test_set = np.array(test_set)

time_steps = 5
y_column = 1

def split_xy(dataset, time_steps, y_column):                 
    x = []
    y = []
    for i in range(len(dataset)):
        x_end_number = i + time_steps      # 0 + 5   > 5
        y_end_number = x_end_number + y_column - 1    # 5 + 3 -1 > 7
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : -1]                   # 0 : 5 , : -1   > 0행~4행, 마지막열 뺀 전부
        tmp_y = dataset[x_end_number-1 : y_end_number, -1]       # 5 - 1 : 7 , -1  > 마지막 열의 4~6행
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_train_set, y_train_set = split_xy(train_set, time_steps, y_column)



x_train, x_test, y_train, y_test = train_test_split(x_train_set, y_train_set, train_size=0.7, shuffle=False)



print(x_train.shape) # (7011, 5, 4)
print(y_train.shape) # (7011, 1)
print(x_test.shape)
print(y_test.shape)

x_train = x_train.reshape(24541, 20)
x_test = x_test.reshape(10518, 20)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
















bayesian_params = {
    'colsample_bytree' : (0.5, 0.7),
    'max_depth' : (10,18),
    'min_child_weight' : (30, 35),
    'reg_alpha' : (43, 47),
    'reg_lambda' : (0.001, 0.01),
    'subsample' : (0.4, 0.7)
}

def xgb_function(max_depth, min_child_weight,subsample, colsample_bytree, reg_lambda,reg_alpha):
    params ={
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                    # 정수만
        'min_child_weight' : int(round(min_child_weight)),
        'subsample' : max(min(subsample,1),0),                  # 0~1 사이값만
        'colsample_bytree' : max(min(colsample_bytree,1),0),
        'reg_lambda' : max(reg_lambda,0),                       # 양수만
        'reg_alpha' : max(reg_alpha,0),
    }
            
            
        # 'num_leaves':int(round(num_leaves)),
        # 'min_child_samples':int(round(min_child_samples)),
        # 'min_child_weight':int(round(min_child_weight)), # 무조건 정수
        # 'subsample':max(min(subsample, 1), 0),  # 0~1 사이의 값이 들어와야 한다 1이상이면 1
        # 'colsample_bytree':max(min(colsample_bytree, 1), 0),
        # 'max_bin':max(int(round(max_bin)), 10), # 10이상의 정수
        # 'reg_lambda':max(reg_lambda, 0),        # 무조건 양수만
        # 'reg_alpha':max(reg_alpha, 0)           # 무조건 양수만
    
    # *여러개의인자를받겠다     
    # **키워드받겠다(딕셔너리형태)
    
    model = XGBRegressor(**params) # 모델을 함수안에 써야한다
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='rmse',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    result = r2_score(y_test, y_predict)
    
    return result

lgb_bo = BayesianOptimization(f=xgb_function,
                              pbounds=bayesian_params,
                              random_state=1234)

lgb_bo.maximize(init_points=5, n_iter=50)

print(lgb_bo.max)
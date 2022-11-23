import pandas as pd
import datetime as dt
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Data
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)

path = 'D:/study_data/_data/aifactory/Task_01/data_city/'
train_set = pd.read_csv(path + 'data_tr_city.csv')
test_set = pd.read_csv(path + 'data_ts_city.csv')


train_set['datetime'] = pd.to_datetime(train_set['datetime'])
train_set['연도별'] = train_set['datetime'].dt.year
train_set['월별'] = train_set['datetime'].dt.month
train_set['일별'] = train_set['datetime'].dt.day
train_set['시간별'] = train_set['datetime'].dt.hour
# train_set['요일별'] = 

train_set = train_set.drop(['datetime'], axis=1)
train_set = train_set[['연도별', '월별', '일별', '시간별', '구미 혁신도시배수지 유출유량 적산차']]



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
train_set['구미 혁신도시배수지 유출유량 적산차'] = train_set['구미 혁신도시배수지 유출유량 적산차'].fillna(train_set['구미 혁신도시배수지 유출유량 적산차'].mean())
test_set['구미 혁신도시배수지 유출유량 적산차'] = test_set['구미 혁신도시배수지 유출유량 적산차'].fillna(test_set['구미 혁신도시배수지 유출유량 적산차'].mean())

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



n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state = 123)

parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000],
              'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
              'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100],
              'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10],
              'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
              'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] ,
              'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10],
              'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10]
              }



# https://xgboost.readthedocs.io/en/stable/parameter.html
#2.모델 

xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id='0')

model = XGBRegressor(n_estimators=1000, tree_method='gpu_hist', predictor='gpu_predictor', gpu_id='0')


# xgb = XGBRegressor(random_state = 123)

# model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id='0')
model.fit(x_train, y_train)
results = model.score(x_test, y_test)

print(results)


from sklearn.metrics import accuracy_score 
y_predict = model.predict(x_test) # predict는 make_pipeline을 이용하여 scaler가 적용된 상태다
y_test = y_test.reshape(10518,)
print(y_test)
print(y_predict)
# [175. 315. 257. ... 335. 141. 112.]
# [ 241.38544  308.05215  308.08627 ... -178.04787 -196.88449 -213.95715]

'''
with open('D:\_AIA_Team_Project_Data\KOGPT/test.txt', encoding='UTF-8') as f :
    line = f.read()
    
line = "".join([s for s in line.strip().splitlines(True) if s.strip()])
line = line.split('\n')
line = str(line)
line = line.replace("'", '')

print(line)
'''
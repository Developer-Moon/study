from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#----------------------------------------------------------------------------------------#
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다
#----------------------------------------------------------------------------------------#


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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(),
           PowerTransformer(method='yeo-johnson'), PowerTransformer(method='box-cox')]
models = [LGBMRegressor(), CatBoostRegressor(verbose=0), XGBRegressor(), RandomForestRegressor()]

for i in scalers:
    scaler = i
    if str(scaler) == str(PowerTransformer(method='box-cox')): # 이 부분이 좀 이해가 안 간다...
        try:
            x_train = scaler.fit_transform(x_train) 
        except:
            print('진짜 도덕책이네...')
            break

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for model in models:       
        model = model
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        result = r2_score(y_test, y_predict)   
        
        if str(model).startswith('<cat'):
            print('CatBoostClassifier -', str(scaler).replace('()',' :'), round(result, 4))
        elif str(model).startswith('XGB'):
            print('XGBClassifier -', str(scaler).replace('()',' :'), round(result, 4))
        else:
            print(str(model).replace('()',' -'), str(scaler).replace('()',' :'), round(result, 4))    
        
# LGBMRegressor - StandardScaler : 0.9575
# CatBoostClassifier - StandardScaler : 0.959
# XGBClassifier - StandardScaler : 0.9551
# RandomForestRegressor - StandardScaler : 0.9536
# LGBMRegressor - MinMaxScaler : 0.9575
# CatBoostClassifier - MinMaxScaler : 0.959
# XGBClassifier - MinMaxScaler : 0.9552
# RandomForestRegressor - MinMaxScaler : 0.9542
# LGBMRegressor - MaxAbsScaler : 0.9575
# CatBoostClassifier - MaxAbsScaler : 0.959
# XGBClassifier - MaxAbsScaler : 0.9552
# RandomForestRegressor - MaxAbsScaler : 0.9533
# LGBMRegressor - RobustScaler : 0.9575
# CatBoostClassifier - RobustScaler : 0.959
# XGBClassifier - RobustScaler : 0.9551
# RandomForestRegressor - RobustScaler : 0.9542
# LGBMRegressor - QuantileTransformer : 0.9575
# CatBoostClassifier - QuantileTransformer : 0.959
# XGBClassifier - QuantileTransformer : 0.9549
# RandomForestRegressor - QuantileTransformer : 0.9531
# LGBMRegressor - PowerTransformer : 0.9575
# CatBoostClassifier - PowerTransformer : 0.959
# XGBClassifier - PowerTransformer : 0.9549
# RandomForestRegressor - PowerTransformer : 0.9552
# 진짜 도덕책이네...
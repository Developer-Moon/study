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
        
        print(i.__class__.__name__, '-', model.__class__.__name__, ':', round(result, 4))
        
# StandardScaler - LGBMRegressor : 0.9575
# StandardScaler - CatBoostRegressor : 0.959
# StandardScaler - XGBRegressor : 0.9551
# StandardScaler - RandomForestRegressor : 0.9536
# MinMaxScaler - LGBMRegressor : 0.9575
# MinMaxScaler - CatBoostRegressor : 0.959
# MinMaxScaler - XGBRegressor : 0.9552
# MinMaxScaler - RandomForestRegressor : 0.9537
# MaxAbsScaler - LGBMRegressor : 0.9575
# MaxAbsScaler - CatBoostRegressor : 0.959
# MaxAbsScaler - XGBRegressor : 0.9552
# MaxAbsScaler - RandomForestRegressor : 0.9543
# RobustScaler - LGBMRegressor : 0.9575
# RobustScaler - CatBoostRegressor : 0.959
# RobustScaler - XGBRegressor : 0.9551
# RobustScaler - RandomForestRegressor : 0.9535
# QuantileTransformer - LGBMRegressor : 0.9575
# QuantileTransformer - CatBoostRegressor : 0.959
# QuantileTransformer - XGBRegressor : 0.9549
# QuantileTransformer - RandomForestRegressor : 0.9526
# PowerTransformer - LGBMRegressor : 0.9575
# PowerTransformer - CatBoostRegressor : 0.959
# PowerTransformer - XGBRegressor : 0.9549
# PowerTransformer - RandomForestRegressor : 0.9538
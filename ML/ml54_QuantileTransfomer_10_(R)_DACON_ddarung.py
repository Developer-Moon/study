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
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set = train_set.fillna(0)

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
        
# StandardScaler - LGBMRegressor : 0.794
# StandardScaler - CatBoostRegressor : 0.7999
# StandardScaler - XGBRegressor : 0.7677
# StandardScaler - RandomForestRegressor : 0.7904
# MinMaxScaler - LGBMRegressor : 0.7849
# MinMaxScaler - CatBoostRegressor : 0.7997
# MinMaxScaler - XGBRegressor : 0.7677
# MinMaxScaler - RandomForestRegressor : 0.7901
# MaxAbsScaler - LGBMRegressor : 0.7849
# MaxAbsScaler - CatBoostRegressor : 0.7997
# MaxAbsScaler - XGBRegressor : 0.7677
# MaxAbsScaler - RandomForestRegressor : 0.7788
# RobustScaler - LGBMRegressor : 0.7899
# RobustScaler - CatBoostRegressor : 0.7997
# RobustScaler - XGBRegressor : 0.7682
# RobustScaler - RandomForestRegressor : 0.7859
# QuantileTransformer - LGBMRegressor : 0.7848
# QuantileTransformer - CatBoostRegressor : 0.7988
# QuantileTransformer - XGBRegressor : 0.773
# QuantileTransformer - RandomForestRegressor : 0.7901
# PowerTransformer - LGBMRegressor : 0.7917
# PowerTransformer - CatBoostRegressor : 0.7989
# PowerTransformer - XGBRegressor : 0.772
# PowerTransformer - RandomForestRegressor : 0.788
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
        
        if str(model).startswith('<cat'):
            print('CatBoostClassifier -', str(scaler).replace('()',' :'), round(result, 4))
        elif str(model).startswith('XGB'):
            print('XGBClassifier -', str(scaler).replace('()',' :'), round(result, 4))
        else:
            print(str(model).replace('()',' -'), str(scaler).replace('()',' :'), round(result, 4))    
        
# LGBMRegressor - StandardScaler : 0.794
# CatBoostClassifier - StandardScaler : 0.7999
# XGBClassifier - StandardScaler : 0.7677
# RandomForestRegressor - StandardScaler : 0.7867
# LGBMRegressor - MinMaxScaler : 0.7849
# CatBoostClassifier - MinMaxScaler : 0.7997
# XGBClassifier - MinMaxScaler : 0.7677
# RandomForestRegressor - MinMaxScaler : 0.7891
# LGBMRegressor - MaxAbsScaler : 0.7849
# CatBoostClassifier - MaxAbsScaler : 0.7997
# XGBClassifier - MaxAbsScaler : 0.7677
# RandomForestRegressor - MaxAbsScaler : 0.7853
# LGBMRegressor - RobustScaler : 0.7899
# CatBoostClassifier - RobustScaler : 0.7997
# XGBClassifier - RobustScaler : 0.7682
# RandomForestRegressor - RobustScaler : 0.7892
# LGBMRegressor - QuantileTransformer : 0.7848
# CatBoostClassifier - QuantileTransformer : 0.7988
# XGBClassifier - QuantileTransformer : 0.773
# RandomForestRegressor - QuantileTransformer : 0.7929
# LGBMRegressor - PowerTransformer : 0.7917
# CatBoostClassifier - PowerTransformer : 0.7989
# XGBClassifier - PowerTransformer : 0.772
# RandomForestRegressor - PowerTransformer : 0.7806
# 진짜 도덕책이네...
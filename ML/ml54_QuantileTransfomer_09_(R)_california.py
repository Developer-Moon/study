from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import fetch_california_housing
#----------------------------------------------------------------------------------------#
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다
#----------------------------------------------------------------------------------------#


#1. 데이터
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target

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
        
# LGBMRegressor - StandardScaler : 0.828
# CatBoostClassifier - StandardScaler : 0.8456
# XGBClassifier - StandardScaler : 0.8265
# RandomForestRegressor - StandardScaler : 0.804
# LGBMRegressor - MinMaxScaler : 0.8293
# CatBoostClassifier - MinMaxScaler : 0.8456
# XGBClassifier - MinMaxScaler : 0.8265
# RandomForestRegressor - MinMaxScaler : 0.8073
# LGBMRegressor - MaxAbsScaler : 0.8293
# CatBoostClassifier - MaxAbsScaler : 0.8456
# XGBClassifier - MaxAbsScaler : 0.8265
# RandomForestRegressor - MaxAbsScaler : 0.8053
# LGBMRegressor - RobustScaler : 0.8306
# CatBoostClassifier - RobustScaler : 0.8456
# XGBClassifier - RobustScaler : 0.8264
# RandomForestRegressor - RobustScaler : 0.8061
# LGBMRegressor - QuantileTransformer : 0.8293
# CatBoostClassifier - QuantileTransformer : 0.8456
# XGBClassifier - QuantileTransformer : 0.8258
# RandomForestRegressor - QuantileTransformer : 0.8072
# LGBMRegressor - PowerTransformer : 0.8274
# CatBoostClassifier - PowerTransformer : 0.8456
# XGBClassifier - PowerTransformer : 0.8258
# RandomForestRegressor - PowerTransformer : 0.8043
# 진짜 도덕책이네...
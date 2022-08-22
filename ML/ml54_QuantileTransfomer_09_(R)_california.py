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
        
        print(i.__class__.__name__, '-', model.__class__.__name__, ':', round(result, 4))
        
# StandardScaler - LGBMRegressor : 0.828
# StandardScaler - CatBoostRegressor : 0.8456
# StandardScaler - XGBRegressor : 0.8265
# StandardScaler - RandomForestRegressor : 0.8069
# MinMaxScaler - LGBMRegressor : 0.8293
# MinMaxScaler - CatBoostRegressor : 0.8456
# MinMaxScaler - XGBRegressor : 0.8265
# MinMaxScaler - RandomForestRegressor : 0.8065
# MaxAbsScaler - LGBMRegressor : 0.8293
# MaxAbsScaler - CatBoostRegressor : 0.8456
# MaxAbsScaler - XGBRegressor : 0.8265
# MaxAbsScaler - RandomForestRegressor : 0.805
# RobustScaler - LGBMRegressor : 0.8306
# RobustScaler - CatBoostRegressor : 0.8456
# RobustScaler - XGBRegressor : 0.8264
# RobustScaler - RandomForestRegressor : 0.8046
# QuantileTransformer - LGBMRegressor : 0.8293
# QuantileTransformer - CatBoostRegressor : 0.8456
# QuantileTransformer - XGBRegressor : 0.8258
# QuantileTransformer - RandomForestRegressor : 0.8052
# PowerTransformer - LGBMRegressor : 0.8274
# PowerTransformer - CatBoostRegressor : 0.8456
# PowerTransformer - XGBRegressor : 0.8258
# PowerTransformer - RandomForestRegressor : 0.8042
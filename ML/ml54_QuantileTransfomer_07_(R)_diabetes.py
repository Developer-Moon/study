from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_diabetes
#----------------------------------------------------------------------------------------#
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다
#----------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_diabetes()
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
        
# StandardScaler - LGBMRegressor : 0.3444
# StandardScaler - CatBoostRegressor : 0.3953
# StandardScaler - XGBRegressor : 0.2601
# StandardScaler - RandomForestRegressor : 0.3931
# MinMaxScaler - LGBMRegressor : 0.3228
# MinMaxScaler - CatBoostRegressor : 0.3956
# MinMaxScaler - XGBRegressor : 0.2706
# MinMaxScaler - RandomForestRegressor : 0.4279
# MaxAbsScaler - LGBMRegressor : 0.3234
# MaxAbsScaler - CatBoostRegressor : 0.3956
# MaxAbsScaler - XGBRegressor : 0.2706
# MaxAbsScaler - RandomForestRegressor : 0.4252
# RobustScaler - LGBMRegressor : 0.3343
# RobustScaler - CatBoostRegressor : 0.3955
# RobustScaler - XGBRegressor : 0.2672
# RobustScaler - RandomForestRegressor : 0.4153
# QuantileTransformer - LGBMRegressor : 0.3244
# QuantileTransformer - CatBoostRegressor : 0.3951
# QuantileTransformer - XGBRegressor : 0.263
# QuantileTransformer - RandomForestRegressor : 0.4278
# PowerTransformer - LGBMRegressor : 0.3393
# PowerTransformer - CatBoostRegressor : 0.3986
# PowerTransformer - XGBRegressor : 0.2638
# PowerTransformer - RandomForestRegressor : 0.433
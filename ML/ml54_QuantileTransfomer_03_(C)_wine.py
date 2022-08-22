from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_wine
#----------------------------------------------------------------------------------------#
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다
#----------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(),
           PowerTransformer(method='yeo-johnson'), PowerTransformer(method='box-cox')]
models = [LGBMClassifier(), CatBoostClassifier(verbose=0), XGBClassifier(), RandomForestClassifier()]

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
        result = accuracy_score(y_test, y_predict)   
        
        print(i.__class__.__name__, '-', model.__class__.__name__, ':', round(result, 4))
        
# StandardScaler - LGBMClassifier : 0.9444
# StandardScaler - CatBoostClassifier : 0.9444
# StandardScaler - XGBClassifier : 0.8889
# StandardScaler - RandomForestClassifier : 0.9444
# MinMaxScaler - LGBMClassifier : 0.9444
# MinMaxScaler - CatBoostClassifier : 0.9444
# MinMaxScaler - XGBClassifier : 0.8889
# MinMaxScaler - RandomForestClassifier : 0.9444
# MaxAbsScaler - LGBMClassifier : 0.9444
# MaxAbsScaler - CatBoostClassifier : 0.9444
# MaxAbsScaler - XGBClassifier : 0.8889
# MaxAbsScaler - RandomForestClassifier : 0.9444
# RobustScaler - LGBMClassifier : 0.9444
# RobustScaler - CatBoostClassifier : 0.9444
# RobustScaler - XGBClassifier : 0.8889
# RobustScaler - RandomForestClassifier : 0.9444
# QuantileTransformer - LGBMClassifier : 0.9444
# QuantileTransformer - CatBoostClassifier : 0.9444
# QuantileTransformer - XGBClassifier : 0.8889
# QuantileTransformer - RandomForestClassifier : 0.9444
# PowerTransformer - LGBMClassifier : 0.9444
# PowerTransformer - CatBoostClassifier : 0.9444
# PowerTransformer - XGBClassifier : 0.8889
# PowerTransformer - RandomForestClassifier : 0.9444
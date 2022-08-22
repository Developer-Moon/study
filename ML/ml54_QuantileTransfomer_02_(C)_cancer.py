from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer
#----------------------------------------------------------------------------------------#
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다
#----------------------------------------------------------------------------------------#


#1. 데이터
datasets = load_breast_cancer()
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
        
# StandardScaler - LGBMClassifier : 0.9386
# StandardScaler - CatBoostClassifier : 0.9386
# StandardScaler - XGBClassifier : 0.9386
# StandardScaler - RandomForestClassifier : 0.9123
# MinMaxScaler - LGBMClassifier : 0.9298
# MinMaxScaler - CatBoostClassifier : 0.9386
# MinMaxScaler - XGBClassifier : 0.9386
# MinMaxScaler - RandomForestClassifier : 0.9123
# MaxAbsScaler - LGBMClassifier : 0.9298
# MaxAbsScaler - CatBoostClassifier : 0.9386
# MaxAbsScaler - XGBClassifier : 0.9386
# MaxAbsScaler - RandomForestClassifier : 0.9035
# RobustScaler - LGBMClassifier : 0.9386
# RobustScaler - CatBoostClassifier : 0.9386
# RobustScaler - XGBClassifier : 0.9386
# RobustScaler - RandomForestClassifier : 0.9298
# QuantileTransformer - LGBMClassifier : 0.9298
# QuantileTransformer - CatBoostClassifier : 0.9386
# QuantileTransformer - XGBClassifier : 0.9386
# QuantileTransformer - RandomForestClassifier : 0.9386
# PowerTransformer - LGBMClassifier : 0.9386
# PowerTransformer - CatBoostClassifier : 0.9386
# PowerTransformer - XGBClassifier : 0.9386
# PowerTransformer - RandomForestClassifier : 0.9298
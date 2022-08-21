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
        
        if str(model).startswith('<cat'):
            print('CatBoostClassifier -', str(scaler).replace('()',' :'), round(result, 4))
        elif str(model).startswith('XGB'):
            print('XGBClassifier -', str(scaler).replace('()',' :'), round(result, 4))
        else:
            print(str(model).replace('()',' -'), str(scaler).replace('()',' :'), round(result, 4))    
        
# LGBMClassifier - StandardScaler : 0.9386
# CatBoostClassifier - StandardScaler : 0.9386
# XGBClassifier - StandardScaler : 0.9386
# RandomForestClassifier - StandardScaler : 0.9386
# LGBMClassifier - MinMaxScaler : 0.9298
# CatBoostClassifier - MinMaxScaler : 0.9386
# XGBClassifier - MinMaxScaler : 0.9386
# RandomForestClassifier - MinMaxScaler : 0.9298
# LGBMClassifier - MaxAbsScaler : 0.9298
# CatBoostClassifier - MaxAbsScaler : 0.9386
# XGBClassifier - MaxAbsScaler : 0.9386
# RandomForestClassifier - MaxAbsScaler : 0.9298
# LGBMClassifier - RobustScaler : 0.9386
# CatBoostClassifier - RobustScaler : 0.9386
# XGBClassifier - RobustScaler : 0.9386
# RandomForestClassifier - RobustScaler : 0.9123
# LGBMClassifier - QuantileTransformer : 0.9298
# CatBoostClassifier - QuantileTransformer : 0.9386
# XGBClassifier - QuantileTransformer : 0.9386
# RandomForestClassifier - QuantileTransformer : 0.9211
# LGBMClassifier - PowerTransformer : 0.9386
# CatBoostClassifier - PowerTransformer : 0.9386
# XGBClassifier - PowerTransformer : 0.9386
# RandomForestClassifier - PowerTransformer : 0.9035
# 진짜 도덕책이네...
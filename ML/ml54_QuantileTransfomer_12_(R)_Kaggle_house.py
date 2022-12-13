from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from tqdm import tqdm_notebook
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#----------------------------------------------------------------------------------------#
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다
#----------------------------------------------------------------------------------------#


#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)cc

drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] 

test_set.drop(drop_cols, axis = 1, inplace =True)
train_set.drop(drop_cols, axis = 1, inplace =True)

cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

train_set = train_set.fillna(train_set.mean()) 
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['SalePrice'], axis=1)
y = train_set['SalePrice']

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
        
# StandardScaler - LGBMRegressor : 0.8785
# StandardScaler - CatBoostRegressor : 0.9003
# StandardScaler - XGBRegressor : 0.8648
# StandardScaler - RandomForestRegressor : 0.8695
# MinMaxScaler - LGBMRegressor : 0.8815
# MinMaxScaler - CatBoostRegressor : 0.9003
# MinMaxScaler - XGBRegressor : 0.8647
# MinMaxScaler - RandomForestRegressor : 0.8688
# MaxAbsScaler - LGBMRegressor : 0.8815
# MaxAbsScaler - CatBoostRegressor : 0.9003
# MaxAbsScaler - XGBRegressor : 0.8647
# MaxAbsScaler - RandomForestRegressor : 0.8635
# RobustScaler - LGBMRegressor : 0.8774
# RobustScaler - CatBoostRegressor : 0.9003
# RobustScaler - XGBRegressor : 0.8666
# RobustScaler - RandomForestRegressor : 0.8678
# QuantileTransformer - LGBMRegressor : 0.8813
# QuantileTransformer - CatBoostRegressor : 0.9003
# QuantileTransformer - XGBRegressor : 0.8743
# QuantileTransformer - RandomForestRegressor : 0.8648
# PowerTransformer - LGBMRegressor : 0.8776
# PowerTransformer - CatBoostRegressor : 0.9018
# PowerTransformer - XGBRegressor : 0.8794
# PowerTransformer - RandomForestRegressor : 0.8657
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import SVR
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np


#1. 데이터
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=68)

           
n_splits = 5              
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)
                    
                                      
#2. 모델구성
model = SVR()


#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold)               # cv=5 라면 kfold를 5로 쓴다
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_predict)

print('r2 :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)
print(y_predict)
print('cross_val_predict r2 : ', r2)

# r2 : [-0.04102167 -0.00920767 -0.06124829 -0.0770459  -0.13418722] 
# cross_val_score : -0.0645
# cross_val_predict r2 :  -0.044926941270609255
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression # 회귀, 이진분류
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from tqdm import tqdm_notebook
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


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

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)



#2. 모델구성
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)



#3 훈련, 결과, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('일반 score :', model.score(x_test, y_test)) 
print('CV :', scores)
print('CV 엔빵 :', np.mean(scores))








# PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias=False 첫번째 컬럼 1을 안나오게 한다??
xp = pf.fit_transform(x)
print(xp.shape) # (150, 14)

x_train, x_test, y_train, y_test = train_test_split(xp, y, train_size=0.8, random_state=1234)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델구성
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')

print('폴리 score :', model.score(x_test, y_test))
print('폴리CV :', scores)
print('폴리CV 엔빵 :', np.mean(scores))

# 일반 score : -1.3005594486969019e+21
# CV : [0.85769281 0.84391285 0.82917826 0.77894992 0.12877971]
# CV 엔빵 : 0.6877027078330764
# (1460, 2925)
# 폴리 score : 0.1511659706154863
# 폴리CV : [0.57125648 0.5656226  0.32975685 0.06230797 0.55447872]
# 폴리CV 엔빵 : 0.41668452387510635
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tqdm import tqdm_notebook
import pandas as pd
import time


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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
'''
랜덤포레스트의 파라미터들!
parameters = [
    {'n_estimators':[100,200]},
    {'max_depth':[6,8,10,12]},
    {'min_samples_leaf':[3,5,7,10]},
    {'min_samples_split':[2,3,5,10]},
    {'n_jobs':[-1,2,4]}
]
'''       


#2. 모델구성
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1)


# 3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()

ypred = model.predict(x_test)
ypred_best = model.best_estimator_.predict(x_test)

print('최적의 매개변수 :', model.best_estimator_)
print('최적의 파라미터 :', model.best_params_)
print('best_score :', model.best_score_)
print('model.score :', model.score(x_test, y_test))
print('r2 score :', r2_score(y_test, ypred))
print('best tuned r2 :', r2_score(y_test, ypred_best))
print('걸린시간 :', round(end-start,2),'초')

# Fitting 5 folds for each of 152 candidates, totalling 760 fits
# 최적의 매개변수 : RandomForestRegressor(max_depth=12, n_jobs=4)
# 최적의 파라미터 : {'max_depth': 12, 'n_estimators': 100, 'n_jobs': 4}      
# best_score : 0.8470119678738655
# model.score : 0.8898475987516832
# r2 score : 0.8898475987516832
# best tuned r2 : 0.8898475987516832
# 걸린시간 : 91.84 초
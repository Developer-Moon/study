import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import math
import time
import seaborn as sns
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE                                     

# 1. 데이터
path  = './_data/dacon_travel/'      
train = pd.read_csv(path + 'train.csv', index_col=0)
test  = pd.read_csv(path + 'test.csv', index_col=0)


print(train.info())
'''
0   Age                       1861 non-null   float64
 1   TypeofContact             1945 non-null   object
 2   CityTier                  1955 non-null   int64
 3   DurationOfPitch           1853 non-null   float64
 4   Occupation                1955 non-null   object
 5   Gender                    1955 non-null   object
 6   NumberOfPersonVisiting    1955 non-null   int64
 7   NumberOfFollowups         1942 non-null   float64
 8   ProductPitched            1955 non-null   object
 9   PreferredPropertyStar     1945 non-null   float64
 10  MaritalStatus             1955 non-null   object
 11  NumberOfTrips             1898 non-null   float64
 12  Passport                  1955 non-null   int64
 13  PitchSatisfactionScore    1955 non-null   int64
 14  OwnCar                    1955 non-null   int64
 15  NumberOfChildrenVisiting  1928 non-null   float64
 16  Designation               1955 non-null   object
 17  MonthlyIncome             1855 non-null   float64
 18  ProdTaken                 1955 non-null   int64
 '''

le = LabelEncoder() 
train_cols = np.array(train.columns)

for i in train_cols:
    if train[i].dtype == 'object':
        train[i] = le.fit_transform(train[i])
        test[i] = le.fit_transform(test[i])


# sns.set(font_scale= 0.8 )
# sns.heatmap(data=train.corr(), square= True, annot=True, cbar=True) # square: 정사각형, annot: 안에 수치들 ,cbar: 옆에 bar
# plt.show() 

train = train.dropna()

test['DurationOfPitch'].fillna(test['DurationOfPitch'].mean(), inplace=True)
test['NumberOfFollowups'].fillna(test['NumberOfFollowups'].mean(), inplace=True)
test['PreferredPropertyStar'].fillna(test['PreferredPropertyStar'].median(), inplace=True)
test['NumberOfTrips'].fillna(test['NumberOfTrips'].median(), inplace=True)



x_pd = train.drop(['Age','ProductPitched', 'MonthlyIncome', 'ProdTaken', 'NumberOfChildrenVisiting'], axis=1)
y_pd = train['ProdTaken']
test = test.drop(['Age', 'ProductPitched', 'MonthlyIncome', 'NumberOfChildrenVisiting'], axis=1)

x = np.array(x_pd)
y = np.array(y_pd)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=702, stratify=y)



'''
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위: ', quartile_1)
    print('q2: ', q2)
    print('3사분위: ', quartile_3)
    iqr = quartile_3-quartile_1 # interquartile range
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print(upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

def outliers_printer(dataset):
    plt.figure(figsize=(10,13))
    for i in range(dataset.shape[1]):
        col = dataset[:, i]
        outliers_loc = outliers(col)
        print(i, '열의 이상치의 위치: ', outliers_loc, '\n')
        plt.subplot(math.ceil(dataset.shape[1]/2),2,i+1)
        plt.boxplot(col)
        plt.title(i)
        
    plt.show()
outliers_printer(x)
'''

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=9)

parameters = [
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'n_jobs':[-1,2,4]},
    {'max_depth':[6,8,10,12], 'min_samples_leaf':[3,5,7,10], 'n_jobs':[-1,2,4]},
    {'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10], 'n_jobs':[-1,2,4]},
    {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
    ]                                                                                       
                                                                                                        
                                                                                                        
#2. 모델구성
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1, refit=True, n_jobs=-1, n_iter=15)



#3. 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score :', acc)

# print("________________________")
# print(model,':',model.feature_importances_) # 

#  [0.03703542 0.04461664 0.0689444  0.03657313 0.05336586 0.03205258 0.03270744 0.05974615 0.10627392 0.04411111 0.08781915 0.04463127
#  0.16393435 0.04281751 0.04478906 0.02729187 0.03262447 0.04066554]
 
# 5. 제출 준비
y_submit = model.predict(test)

submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
submission['ProdTaken'] = y_submit
submission.to_csv(path + 'sample_submission2.csv', index = True)




# model.score : 0.8900255754475703
# accuracy_score : 0.8900255754475703

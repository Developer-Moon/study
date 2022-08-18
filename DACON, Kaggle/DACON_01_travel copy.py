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


import warnings
warnings.filterwarnings(action='ignore')

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

'''
sns.set(font_scale= 0.8 )
sns.heatmap(data=train.corr(), square= True, annot=True, cbar=True) # square: 정사각형, annot: 안에 수치들 ,cbar: 옆에 bar
plt.show() 

Age = -0.14
ProductPitched = -0.15
MonthlyIncome  = -0.14
'''

train = train.dropna()
train.drop(['ProdTaken', 'Age', 'ProductPitched'], axis=1, inplace=True)
test.drop(['Age', 'ProductPitched'], axis=1, inplace=True)

print(test.isnull().sum())


test['DurationOfPitch'].fillna(test['DurationOfPitch'].mean(), inplace=True) 
test['NumberOfFollowups'].fillna(test['NumberOfFollowups'].mean(), inplace=True) 
test['PreferredPropertyStar'].fillna(test['PreferredPropertyStar'].mean(), inplace=True) 
test['NumberOfTrips'].fillna(test['NumberOfTrips'].mean(), inplace=True) 
test['NumberOfChildrenVisiting'].fillna(test['NumberOfChildrenVisiting'].mean(), inplace=True) 
test['MonthlyIncome'].fillna(test['MonthlyIncome'].mean(), inplace=True) 


train = np.array(train)



def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    
    print('1사분위 :', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr :', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    
    return np.where((data_out>upper_bound) | (data_out<lower_bound)) # 괄호안의 만족하는 것을 반환한다
outliers_loc1 = outliers(test[:,10])
print('이상치의 위치 :', outliers_loc1)


import matplotlib.pyplot as plt
plt.boxplot(outliers_loc1)
plt.show()









'''

x = train.drop(['ProdTaken', 'Age', 'ProductPitched'], axis=1)
# x = train.drop(['ProdTaken', 'Age', 'ProductPitched'], axis=1) model.score : 0.8757575757575757

y = train['ProdTaken']




x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=702, stratify=y)

                                                                                
                                                                                                        
                                                                                                        
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






 
# 5. 제출 준비
# y_submit = model.predict(test)

# submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
# submission['ProdTaken'] = y_submit
# submission.to_csv(path + 'sample_submission2.csv', index = True)




# model.score : 0.8900255754475703
# accuracy_score : 0.8900255754475703


'''
from time import time
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 

#1. 데이터
path  = './_data/dacon_travel/' 
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
test_set['TypeofContact'].fillna('Self Enquiry', inplace=True)
train_set['Age'].fillna(train_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
test_set['Age'].fillna(test_set.groupby('Designation')['Age'].transform('mean'), inplace=True)
train_set['Age']=np.round(train_set['Age'],0).astype(int)
test_set['Age']=np.round(test_set['Age'],0).astype(int)



train_set['MonthlyIncome'].fillna(train_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
test_set['MonthlyIncome'].fillna(test_set.groupby('Designation')['MonthlyIncome'].transform('mean'), inplace=True)
print(train_set.describe) #(1955, 19)
print(train_set[train_set['MonthlyIncome'].notnull()].groupby(['Designation'])['MonthlyIncome'].mean())

train_set['NumberOfChildrenVisiting'].fillna(train_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
test_set['NumberOfChildrenVisiting'].fillna(test_set.groupby('MaritalStatus')['NumberOfChildrenVisiting'].transform('mean'), inplace=True)
train_set['NumberOfFollowups'].fillna(train_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
test_set['NumberOfFollowups'].fillna(test_set.groupby('NumberOfChildrenVisiting')['NumberOfFollowups'].transform('mean'), inplace=True)
# combine = [train_set,test_set]
# for dataset in combine:    
#     dataset.loc[ dataset['NumberOfChildrenVisiting'] < 1, 'NumberOfChildrenVisiting'] = 0
#     dataset.loc[ dataset['NumberOfChildrenVisiting'] >= 1, 'NumberOfChildrenVisiting'] = 1
# print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())
# print(train_set.isnull().sum()) 

train_set['DurationOfPitch']=train_set['DurationOfPitch'].fillna(0)
test_set['DurationOfPitch']=test_set['DurationOfPitch'].fillna(0)
# train_set['DurationOfPitch'].fillna(train_set.groupby('NumberOfChildrenVisiting')['DurationOfPitch'].transform('mean'), inplace=True)
# test_set['DurationOfPitch'].fillna(test_set.groupby('NumberOfChildrenVisiting')['DurationOfPitch'].transform('mean'), inplace=True)
# print(train_set.isnull().sum()) 


print(train_set[train_set['DurationOfPitch'].notnull()].groupby(['NumberOfChildrenVisiting'])['DurationOfPitch'].mean())


train_set['PreferredPropertyStar'].fillna(train_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
test_set['PreferredPropertyStar'].fillna(test_set.groupby('Occupation')['PreferredPropertyStar'].transform('mean'), inplace=True)
print(train_set[train_set['PreferredPropertyStar'].notnull()].groupby(['ProdTaken'])['PreferredPropertyStar'].mean())


# train_set['AgeBand'] = pd.cut(train_set['Age'], 5)
# 임의로 5개 그룹을 지정
# print(train_set['AgeBand'])
# [(17.957, 26.6] < (26.6, 35.2] < (35.2, 43.8] <
# (43.8, 52.4] < (52.4, 61.0]]
combine = [train_set,test_set]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 26.6, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 26.6) & (dataset['Age'] <= 35.2), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 35.2) & (dataset['Age'] <= 43.8), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 43.8) & (dataset['Age'] <= 52.4), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 52.4, 'Age'] = 4
# train_set = train_set.drop(['AgeBand'], axis=1)
# print(train_set[train_set['NumberOfTrips'].notnull()].groupby(['DurationOfPitch'])['PreferredPropertyStar'].mean())
train_set['NumberOfTrips'].fillna(train_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
test_set['NumberOfTrips'].fillna(test_set.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
# print(train_set[train_set['NumberOfChildrenVisiting'].notnull()].groupby(['MaritalStatus'])['NumberOfChildrenVisiting'].mean())



train_set.loc[ train_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
test_set.loc[ test_set['Gender'] =='Fe Male' , 'Gender'] = 'Female'
cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

# print(train_set)

# print(train_set['TypeofContact'])
def outliers(data_out):
    quartile_1, q2 , quartile_3 = np.percentile(data_out,
                                               [25,50,75]) # percentile 백분위
    print("1사분위 : ",quartile_1) # 25% 위치인수를 기점으로 사이에 값을 구함
    print("q2 : ",q2) # 50% median과 동일 
    print("3사분위 : ",quartile_3) # 75% 위치인수를 기점으로 사이에 값을 구함
    iqr =quartile_3-quartile_1  # 75% -25%
    print("iqr :" ,iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound)|
                    (data_out<lower_bound))
                     
                           
# print(train_set['Designation'].unique())

# Age_out_index= outliers(train_set['Age'])[0]
# TypeofContact_out_index= outliers(train_set['TypeofContact'])[0] # 0
# CityTier_out_index= outliers(train_set['CityTier'])[0] # 0
DurationOfPitch_out_index= outliers(train_set['DurationOfPitch'])[0] #44
Gender_out_index= outliers(train_set['Gender'])[0] # 0
NumberOfPersonVisiting_out_index= outliers(train_set['NumberOfPersonVisiting'])[0] # 1
NumberOfFollowups_out_index= outliers(train_set['NumberOfFollowups'])[0] # 0
ProductPitched_index= outliers(train_set['ProductPitched'])[0] # 0
PreferredPropertyStar_out_index= outliers(train_set['PreferredPropertyStar'])[0]  # 0
MaritalStatus_out_index= outliers(train_set['MaritalStatus'])[0] # 0
NumberOfTrips_out_index= outliers(train_set['NumberOfTrips'])[0] # 38
Passport_out_index= outliers(train_set['Passport'])[0] # 0
PitchSatisfactionScore_out_index= outliers(train_set['PitchSatisfactionScore'])[0] # 0
OwnCar_out_index= outliers(train_set['OwnCar'])[0] # 0
NumberOfChildrenVisiting_out_index= outliers(train_set['NumberOfChildrenVisiting'])[0] # 0
Designation_out_index= outliers(train_set['Designation'])[0] # 89
MonthlyIncome_out_index= outliers(train_set['MonthlyIncome'])[0] # 138

lead_outlier_index = np.concatenate((#Age_out_index,                            # acc : 0.8650306748466258
                                    #  TypeofContact_out_index,                 # acc : 0.8920454545454546
                                    #  CityTier_out_index,                      # acc : 0.8920454545454546
                                     DurationOfPitch_out_index,               # acc : 0.9156976744186046
                                    #  Gender_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfPersonVisiting_out_index,        # acc : 0.8835227272727273
                                    #  NumberOfFollowups_out_index,             # acc : 0.8942598187311178
                                    #  ProductPitched_index,                    # acc : 0.8920454545454546
                                    #  PreferredPropertyStar_out_index,         # acc : 0.8920454545454546
                                    #  MaritalStatus_out_index,                 # acc : 0.8920454545454546
                                    #  NumberOfTrips_out_index,                 # acc : 0.8670520231213873
                                    #  Passport_out_index,                      # acc : 0.8920454545454546
                                    #  PitchSatisfactionScore_out_index,        # acc : 0.8920454545454546
                                    #  OwnCar_out_index,                        # acc : 0.8920454545454546
                                    #  NumberOfChildrenVisiting_out_index,      # acc : 0.8920454545454546
                                    #  Designation_out_index,                   # acc : 0.8869047619047619
                                    #  MonthlyIncome_out_index                  # acc : 0.8932926829268293
                                     ),axis=None)
print(len(lead_outlier_index)) #577

lead_not_outlier_index = []
for i in train_set.index:
    if i not in lead_outlier_index :
        lead_not_outlier_index.append(i)
train_set_clean = train_set.loc[lead_not_outlier_index]      
train_set_clean = train_set_clean.reset_index(drop=True)
# print(train_set_clean)
x = train_set_clean.drop(['ProdTaken',
                          'NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                          'Designation'
                          ], axis=1)
# x = train_set_clean.drop(['ProdTaken'], axis=1)
test_set = test_set.drop(['NumberOfChildrenVisiting',
                          'NumberOfPersonVisiting',
                          'OwnCar', 
                          'MonthlyIncome', 
                          'NumberOfFollowups',
                          'Designation'
                          ], axis=1)
y = train_set_clean['ProdTaken']
print(x.shape)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.93,shuffle=True,random_state=1234,stratify=y)

from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
# 2. 모델

n_splits = 6
# 최상의 점수 :  0.9044520547945205
# acc : 0.954248366013072
# 걸린 시간 : 5.827547073364258 
kfold = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=123)

cat_paramets = {"learning_rate" : [0.20909079092170735],
                'depth' : [8],
                'od_pval' : [0.236844398775451],
                'model_size_reg': [0.30614059763442997],
                'l2_leaf_reg' :[5.535171839105427]}
cat = CatBoostClassifier(random_state=123,verbose=False,n_estimators=500)
model = RandomizedSearchCV(cat,cat_paramets,cv=kfold,n_jobs=-1)

import time 
start_time = time.time()
model.fit(x_train,y_train)   
end_time = time.time()-start_time 
y_predict = model.predict(x_test)
results = accuracy_score(y_test,y_predict)
print('최적의 매개변수 : ',model.best_params_)
print('최상의 점수 : ',model.best_score_)
print('acc :',results)
print('걸린 시간 :',end_time)

# model.fit(x,y)
# y_summit = model.predict(test_set)
# y_summit = np.round(y_summit,0)
# submission = pd.read_csv(path + 'sample_submission.csv',#예측에서 쓸거야!!
#                       )
# submission['ProdTaken'] = y_summit

# submission.to_csv(path + 'sample_submission2.csv',index=False)


# submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
# submission['ProdTaken'] = y_submit
# submission.to_csv(path + 'sample_submission2.csv', index = True)
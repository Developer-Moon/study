from cProfile import label
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.covariance import EllipticEnvelope
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # simple linear
import seaborn as sns


path = './_data/ddarung/'                                       
train_set = pd.read_csv(path + 'train.csv', index_col=0)                                                               
test_set = pd.read_csv(path + 'test.csv', index_col=0) 

train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   
          
print(train_set.shape) # (1459, 10)  
print(train_set.isnull().sum())  
print(test_set.isnull().sum())  


train_set = np.array(train_set)
                                                                                           
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
outliers_loc_train1 = outliers(train_set[:,7])
outliers_loc_train2 = outliers(train_set[:,8])

print('이상치의 위치 :', outliers_loc_train1)
print('이상치의 위치 :', outliers_loc_train2)

outliers_loc1 = np.array(outliers_loc_train1)
outliers_loc2 = np.array(outliers_loc_train2)

print(outliers_loc1.shape)  #(1, 59)
print(outliers_loc2.shape)

outliers_loc = np.concatenate((outliers_loc1, outliers_loc2), axis=1)
print(outliers_loc.shape) # (1, 147)


result_A = list(np.unique(outliers_loc)) # 중복된 값 없애기

print(result_A)

print(train_set.shape) # (1459, 10)

train_set = pd.DataFrame(train_set)

train_set = train_set.drop(labels=result_A,axis=0,errors='ignore')



print(train_set.shape) # (1322, 10)


train_set = pd.DataFrame([['0':'hour'], '1':'hour_bef_temperature', '2':'hour_bef_precipitation', '3':'hour_bef_windspeed', '4':'hour_bef_humidity',
                          '5':'hour_bef_visibility', '6':'hour_bef_ozone', '7':'hour_bef_pm10', '8':'hour_bef_pm2.5', '9':'count'])
print(train_set.isnull().sum()) 

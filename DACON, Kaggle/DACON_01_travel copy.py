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
                                     

# 1. 데이터
path  = './_data/dacon_travel/'      
train = pd.read_csv(path + 'train.csv', index_col=0)
test  = pd.read_csv(path + 'test.csv', index_col=0)


print(train.describe())

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
'''


train = train.dropna()

test['DurationOfPitch'].fillna(test['DurationOfPitch'].mean(), inplace=True)
test['NumberOfFollowups'].fillna(test['NumberOfFollowups'].mean(), inplace=True)
test['PreferredPropertyStar'].fillna(test['PreferredPropertyStar'].median(), inplace=True)
test['NumberOfTrips'].fillna(test['NumberOfTrips'].median(), inplace=True)
print(test.isnull().sum())
                      
x = train.drop(['ProdTaken', 'Age','MonthlyIncome', 'NumberOfChildrenVisiting', 'TypeofContact'], axis=1)
y = train['ProdTaken']
test = test.drop(['Age','MonthlyIncome', 'NumberOfChildrenVisiting', 'TypeofContact'], axis=1)

x = np.array(x)


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

print(x.shape) # (1649, 14)
x = np.delete(x, 407, 336,  7,   26,   32,   43,   67,   71,   89,  105,  123,  124,  141,
        147,  169,  176,  191,  208,  230,  266,  274,  280,  288,  296,
        299,  310,  348,  353,  364,  387,  416,  418,  431,  483,  511,
        528,  579,  598,  601,  604,  611,  633,  636,  650,  661,  683,
        694,  704,  718,  731,  735,  759,  767,  796,  810,  818,  819,
        838,  850,  852,  861,  864,  896,  912,  915,  931,  932,  951,
        969,  998, 1021, 1047, 1052, 1089, 1096, 1112, 1119, 1131, 1137,
       1141, 1154, 1156, 1165, 1174, 1182, 1200, 1213, 1271, 1313, 1340,
       1345, 1352, 1355, 1364, 1373, 1382, 1415, 1450, 1456, 1458, 1480,
       1490, 1536, 1540, 1564, 1569, 1582, 1590, 1623, 1626, 74,   88,  105,  133,  146,  204,  248,  249,  257,  282,  352,
        463,  522,  587,  642,  733,  779,  831,  892,  911,  991, 1121,
       1152, 1184, 1203, 1280, 1289, 1345, 1356, 1403, 1463, 1479, 1482,
       1492, 1543, 1557, 1565, 1629, 49,   87,  154,  167,  180,  279,  308,  372,  423,  480,  538,
        554,  612,  685,  714,  715,  766,  794,  813,  839,  930,  942,
        951,  952, 1109, 1114, 1147, 1155, 1273, 1296, 1325, 1408, 1480,
       1506, 1513, 1579, axis=0)
print(x.shape)

train_set = train_set.drop(labels=result_A,axis=0,errors='ignore')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=702)


#2. 모델구성
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)

# HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
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
# y_submit = model.predict(test)

# submission = pd.read_csv(path+'sample_submission.csv', index_col=0)
# submission['ProdTaken'] = y_submit
# submission.to_csv(path + 'sample_submission2.csv', index = True)
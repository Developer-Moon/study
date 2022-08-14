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

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    HalvingRandomSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

                                     

# 1. 데이터
path  = './_data/dacon_travel/'      
train = pd.read_csv(path + 'train.csv', index_col=0)
test  = pd.read_csv(path + 'test.csv', index_col=0)

# print(train.info())
# print(test.info())  # ProdTaken 없음

train = train.dropna()                      
test = test.dropna()                      




# print(train.info())
le = LabelEncoder()

train['TypeofContact'] = le.fit_transform(train['TypeofContact'])
train['Occupation'] = le.fit_transform(train['Occupation']) 
train['Gender'] = le.fit_transform(train['Gender']) 
train['ProductPitched'] = le.fit_transform(train['ProductPitched']) 
train['MaritalStatus'] = le.fit_transform(train['MaritalStatus']) 
train['Designation'] = le.fit_transform(train['Designation']) 

test['TypeofContact'] = le.fit_transform(test['TypeofContact'])
test['Occupation'] = le.fit_transform(test['Occupation']) 
test['Gender'] = le.fit_transform(test['Gender']) 
test['ProductPitched'] = le.fit_transform(test['ProductPitched']) 
test['MaritalStatus'] = le.fit_transform(test['MaritalStatus']) 
test['Designation'] = le.fit_transform(test['Designation']) 

 
x = train.drop(['ProdTaken', 'NumberOfChildrenVisiting', 'OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3 제거
y = train['ProdTaken']
"""
x = np.array(x_)
y = np.array(y_)
y = y.reshape(-1, 1) # y값 reshape 해야되서 x도 넘파이로 바꿔 훈련하는 것
"""
test = test.drop(['NumberOfChildrenVisiting', 'OwnCar'], axis=1) # 피처임포턴스로 확인한 중요도 낮은 탑3 제거
# test = np.array(test)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)


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
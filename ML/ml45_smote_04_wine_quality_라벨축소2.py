from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns     
from imblearn.over_sampling import SMOTE
     
     
#1 데이터     
datasets = pd.read_csv('./_data/wine/winequality-white.csv', index_col=None, header=0, sep=';') # 에큐러시가 낮은건 분포가 골로르지 않아서
print(datasets.shape)      # (4898, 12)
print(datasets.describe()) # std = 표준편차
print(datasets.info())     # 결측지X - quality가 float라면 int로 바꿔주던가 생각을 해야한다


datasets2 = datasets.values
print(type(datasets))
print(datasets2.shape)

x = datasets2[:, :11]    # 0컬럼부터 10컬럼까지
y = datasets2[:, 11]     # 
print(x.shape, y.shape) # (4898, 11) (4898,)

print(np.unique(y, return_counts=True))   # numpy에서 유니크 보는법 (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(datasets['quality'].value_counts()) # pandas에서 유니크 보는법

print(y[:20]) # [6. 6. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 7. 5. 7. 6. 8. 6. 5.]

for index, value in enumerate(y): 
    if value == 9 :
     y[index] = 7
    elif value == 8 :
       y[index] = 7
    elif value == 7 :
       y[index] = 7
    elif value == 6 :
       y[index] = 6
    elif value == 5 :
       y[index] = 5
    elif value == 4 :
       y[index] = 4
    elif value == 3 :
       y[index] = 4
    else:
        y[index] = 0
       
print(np.unique(y, return_counts=True)) # (array([4., 5., 6., 7.]), array([ 183, 1457, 2198, 1060], dtype=int64))
    
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)


print(pd.Series(y_train).value_counts())

print('___________________ smote사용 후 ___________________')
smote = SMOTE(random_state=123, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())



#2. 모델
model = RandomForestClassifier() # Lo머시기는 다중분류처럼 생겼는데 2진불류에만 쓰인다 찾아보자


#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)



from sklearn.metrics import accuracy_score, f1_score
print('model.score :', score)
print('acc_score :', accuracy_score(y_test, y_predict))
print('f1_score(macro) :', f1_score(y_test, y_predict, average='macro')) # 다중분류에서 쓰기위해 average='macro'사용
print('f1_score(micro) :', f1_score(y_test, y_predict, average='micro')) # 




# 일반
# acc_score : 0.7316326530612245
# f1_score(macro) : 0.6146535485053382


# 스모트 k_neighbors=1
# acc_score : 0.713265306122449
# f1_score(macro) : 0.6639009169170274

# 스모트 k_neighbors=2
# acc_score : 0.7051020408163265
# f1_score(macro) : 0.6617231543511721


# 스모트 k_neighbors=3
# acc_score : 0.7112244897959183
# f1_score(macro) : 0.6547974805435528

# 스모트 k_neighbors=4
# acc_score : 0.713265306122449
# f1_score(macro) : 0.6578497810801848
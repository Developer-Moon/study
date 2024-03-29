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
import math
# 아웃라이어 확인
# 아웃라이어 처리
# 고고씽
     
     
#1 데이터     
datasets = pd.read_csv('./_data/wine/winequality-white.csv', index_col=None, header=0, sep=';') # 에큐러시가 낮은건 분포가 골로르지 않아서
print(datasets.shape)      # (4898, 12)
print(datasets.describe()) # std = 표준편차
print(datasets.info())     # 결측지X - quality가 float라면 int로 바꿔주던가 생각을 해야한다

# pandas보다 numpy가 계산이 좀 더 빠르다
# datasets = np.array(datasets)

#- pandas를 numpy로 바꾸는법
#1 datasets = datasets.to_numpy()
#2 datasets = np.array(datasets)
#3 datasets = datasets.values
datasets2 = datasets.values
print(type(datasets))
print(datasets2.shape)

x = datasets2[:, :11]    # 0컬럼부터 10컬럼까지
y = datasets2[:, 11]     # 
print(x.shape, y.shape) # (4898, 11) (4898,)

print(np.unique(y, return_counts=True))   # numpy에서 유니크 보는법 (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))
print(datasets['quality'].value_counts()) # pandas에서 유니크 보는법


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
        plt.subplot(math.ceil(dataset.shape[1]/2), 2, i+1)
        plt.boxplot(col)
        plt.title(i)
        
    plt.show()
outliers_printer(x)


x_outliers = [98,  169,  207,  294,  358,  551,  555,  656,  774,  847,  873,]

print(x.shape)
print(len(x_outliers))
x_outliers = list(np.unique(x_outliers))
print(len(x_outliers))
print(x.shape)

x = np.delete(x, x_outliers, axis=0)
y = np.delete(y, x_outliers, axis=0)

print(x.shape)





x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)


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

# model.score : 0.7316326530612245
# acc_score : 0.7316326530612245
# f1_score(macro) : 0.4465744288311123
# f1_score(micro) : 0.7316326530612245


# f1_score 이진분류?? 프리시즌 리콜?

# 그래서 여기서 과제!!! 
# 정밀도와 재현율 합친거?

# 프리시즌과 리콜 이해한걸 선생님한테 보내기
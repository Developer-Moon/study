from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
# pip insatll imblearn
from imblearn.over_sampling import SMOTE 
import pandas as pd
import numpy as np


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)                 # (178, 13) (178,)
print(type(x))                          # <class 'numpy.ndarray'>
print(np.unique(y, return_counts=True)) # (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
print(pd.Series(y).value_counts())      # pandas 
# 1    71
# 0    59
# 2    48

print(y)
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
# 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2

x = x[:-25]
y = y[:-25]
print(pd.Series(y).value_counts())
# 1    71
# 0    59
# 2    18

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)


# 1    57
# 0    47
# 2     6



#2. 모델구성
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score
# print('model.score :', score)
print('acc_score :', accuracy_score(y_test, y_predict))
print('f1_score(macro) :', f1_score(y_test, y_predict, average='macro')) # 다중분류에서 쓰기위해 average='macro'사용
# print('f1_score(micro) :', f1_score(y_test, y_predict, average='micro')) # 

# acc_score : 0.9722222222222222
# f1_score(macro) : 0.9743209876543211

# 데이터 축소 후(2번 라벨 40개 줄인 후)
# acc_score : 0.9666666666666667
# f1_score(macro) : 0.9743209876543211


print('___________________ smote사용 후 ___________________')
smote = SMOTE(random_state=123)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())
# 1    57
# 0    57
# 2    57


#2. 평가
model = RandomForestClassifier()
model.fit(x_train, y_train) #test는 평가라서 train만 훈련

#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score
# print('model.score :', score)
print('acc_score :', accuracy_score(y_test, y_predict))
print('f1_score(macro) :', f1_score(y_test, y_predict, average='macro')) # 다중분류에서 쓰기위해 average='macro'사용
# print('f1_score(micro) :', f1_score(y_test, y_predict, average='micro')) # 


# smote를 하고 1/10으로 나눈다??
# resampling 과정이 오래 걸린다


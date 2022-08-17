from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE # pip insatll imblearn
import pandas as pd
import numpy as np

# label 0 - 357개 -> 112개 삭제해서 재구성
# label 1 - 212개
# smote 비교 [acc, f1_score]


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

# zero = np.array(np.where(y==0))
# zero2 = zero.reshape(212,)[:112]
# # zero3= np.random.choice(zero2,112,replace=False) # 섞고싶으면 !
# x = np.delete(x,zero2,0)
# y = np.delete(y,zero2,0)


# ylist = np.where(y == 0)
# ylist = ylist[0][:112]
# print(ylist)
# y_new = np.delete(y,ylist,axis=0)
# x_new = np.delete(x,ylist,axis=0)




print(pd.Series(y).value_counts())
# 1 : 357
# 0 : 212
y_count = np.array(range(112))
y_112 = []

for index, value in enumerate(y): 
    if value == 0 :
        y_112.append(index)

y_112 = y_112[:112]
y_112 = np.array(y_112)
print(y_112.shape)

print(x.shape) # (569, 30)
print(y.shape) # (569,)
x = np.delete(x, y_112, axis=0)
y = np.delete(y, y_112, axis=0)
print(x.shape) # (457, 30)
print(y.shape) # (457,)

print(pd.Series(y).value_counts())
# 1 : 357
# 0 : 100

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)

print(pd.Series(y_train).value_counts())
# 1    285
# 0    170
smote = SMOTE(random_state=123, k_neighbors=4)
x_train, y_train = smote.fit_resample(x_train, y_train)

print(pd.Series(y_train).value_counts())
# 1    285
# 0    285


#2. 모델구성
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score

print('acc_score :', accuracy_score(y_test, y_predict))
print('f1_score(macro) :', f1_score(y_test, y_predict)) 

# y라벨 축소 후 smote
# acc_score : 0.9782608695652174
# f1_score(macro) : 0.9861111111111112
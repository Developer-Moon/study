from sklearn.experimental import enable_halving_search_cv
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV, KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score 
from sklearn.decomposition import PCA
from keras.datasets import mnist
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# cum_list = [154, 331, 486, 713]

# n_conponent > 0.95 이상
# xgboost, gridSearch or RandomSearch 쓰기

# 27_2 결과보다 높게

# Parameters = [
#     {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.01], "max_depth":[4,5,6]},
#     {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
#     {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.05], "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
# ]
# n_jobs = -1
# tree_method='gpu_hist', predictor='gpu_predictor'
# gpu_id='0'


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
x = x.reshape(70000, 28*28)            # (70000, 784)

pca = PCA(n_components=154)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum =np.cumsum(pca_EVR)
# print(cumsum)
# print(np.argmax(cumsum >= 0.996) + 1)  

x_train = x[:60000] # (60000, 403)
x_test = x[60000:]  # (10000, 403)

parameters = [
    {"n_estimators":[100, 200, 300], "learning_rate":[0.1, 0.3, 0.01], "max_depth":[4,5,6]},
    {"n_estimators":[90, 100, 110], "learning_rate":[0.1, 0.001, 0.01], "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators":[90, 110], "learning_rate":[0.1, 0.001, 0.05], "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
]

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)


#2. 모델구성
model = GridSearchCV(XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id='0'), parameters, cv=kfold, verbose=1, refit=True) 


#3. 컴파일, 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()


#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score :', result)
print('time :', result)
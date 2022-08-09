from cProfile import label
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer, fetch_covtype
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # 즉당히좀 하세여 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape) # (581012, 54)


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

pca = PCA(n_components=20) # 54 -> 10
x = pca.fit_transform(x)

# pca_EVR = pca.explained_variance_ratio_

# cumsum =np.cumsum(pca_EVR)
# print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)

print(np.unique(y_train, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([169472, 226640,  28603,   2198,   7594,  13894,  16408], dtype=int64))


#2. 모델구성
from xgboost import XGBClassifier
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id='0')


#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 :', results)
print('time :', end - start)


# 결과 : 0.8681703570475805
# time : 114.03381443023682

# xgboost - GPU n_components : 10
# 결과 : 0.8406065247885166
# time : 15.830528736114502



# ValueError: Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]
# 라벨 인코더 사용
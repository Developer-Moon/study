from unittest import result
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
#----------------------------------------------------------------------------------------------------------------#
from sklearn.decomposition import PCA  # decomposition 분해
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터.
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (569, 30) (569,)

pca = PCA(n_components=30) # PCA 주성분 분석 - 차원축소(압축) : 열 피쳐
x = pca.fit_transform(x)
print(x.shape)            # (569, 10)



pca_EVR = pca.explained_variance_ratio_ # 변환한 값의 중요도
print(pca_EVR)
'''
[8.05823175e-01 1.63051968e-01 2.13486092e-02 6.95699061e-03
 1.29995193e-03 7.27220158e-04 4.19044539e-04 2.48538539e-04
 8.53912023e-05 3.08071548e-05 6.65623182e-06 1.56778461e-06]
'''
print(sum(pca_EVR)) # 0.9999999203185791 상위 전체의 합 : 1에 가깝다

cumsum = np.cumsum(pca_EVR) # cumsum - 누적합 하나씩 더해가는걸 보여준다
print(cumsum)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()








"""
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)


#2. 모델구성
model = RandomForestRegressor()


#3. 컴파일, 훈련
model.fit(x_train, y_train) # eval_metric='error'


#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 :', results)

# 결과 : 0.7757367060749819

# n_components=11
# 결과 : 0.7896181665270263
"""
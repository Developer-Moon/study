from unittest import result
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
#----------------------------------------------------------------------------------------------------------------#
from sklearn.decomposition import PCA  # decomposition 분해
#----------------------------------------------------------------------------------------------------------------#


#1. 데이터.
datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)   # (506, 13) (506,)

pca = PCA(n_components=12) # PCA 주성분 분석 - 차원축소(압축) : 열 피쳐
x = pca.fit_transform(x)
print(x.shape)            # (506, 2)

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

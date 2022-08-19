from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression # 회귀, 이진분류
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.svm import LinearSVC
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)



#2. 모델구성
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)



#3 훈련, 결과, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')
print('일반 score :', model.score(x_test, y_test)) 
print('CV :', scores)
print('CV 엔빵 :', np.mean(scores))








# PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias=False 첫번째 컬럼 1을 안나오게 한다??
xp = pf.fit_transform(x)
print(xp.shape) # (150, 14)

x_train, x_test, y_train, y_test = train_test_split(xp, y, train_size=0.8, random_state=123)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델구성
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')

print('폴리 score :', model.score(x_test, y_test))
print('폴리CV :', scores)
print('폴리CV 엔빵 :', np.mean(scores))

# (20640, 8) (20640,)
# 일반 score : 0.6065722122106434
# CV : [0.6153114  0.59856063 0.61434956 0.59191848 0.58489428]
# CV 엔빵 : 0.6010068700547342
# (20640, 44)
# 폴리 score : 0.500516568719646
# 폴리CV : [-231.1907676     0.62402233   -3.94169064    0.60985177    0.66558945]
# 폴리CV 엔빵 : -46.646598938385296
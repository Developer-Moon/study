from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression # 회귀, 이진분류
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

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

x_train, x_test, y_train, y_test = train_test_split(xp, y, train_size=0.8, random_state=1234)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델구성
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')

print('폴리 score :', model.score(x_test, y_test))
print('폴리CV :', scores)
print('폴리CV 엔빵 :', np.mean(scores))

# (506, 13) (506,)
# 일반 score : 0.7665382927362877
# CV : [0.71606004 0.67832011 0.65400513 0.56791147 0.7335664 ]
# CV 엔빵 : 0.669972627809433
# (506, 104)
# 폴리 score : 0.8745129304823852
# 폴리CV : [0.7917776  0.8215846  0.79599441 0.81776798 0.81170102]
# 폴리CV 엔빵 : 0.807765121221582
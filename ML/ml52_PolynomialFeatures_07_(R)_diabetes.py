from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes
import numpy as np
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)



#2. 모델구성
model = make_pipeline(MinMaxScaler(), LinearRegression())
model.fit(x_train, y_train)



#3 훈련, 결과, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')

print('Normal score :', model.score(x_test, y_test)) 
print('CV score:', scores)
print('CV score mean :', np.mean(scores))





# PolynomialFeatures -------------------------------------------------------------------------------------------------
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias=False 데이터를 증폭 시킬때 첫번째 열에 1을 안 나오게 한다 
xp = pf.fit_transform(x)                              # 결과에 따라 쓰는 정도의 파라미터로 사용
print(xp.shape)

x_train, x_test, y_train, y_test = train_test_split(xp, y, train_size=0.8, random_state=1234)



#2. 모델구성
model = make_pipeline(MinMaxScaler(), LinearRegression())
model.fit(x_train, y_train)



#3 훈련, 결과, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')

print('PolynomialFeatures score :', model.score(x_test, y_test))
print('PolynomialFeatures CV score :', scores)
print('PolynomialFeatures CV score mean :', np.mean(scores))

# shape - (442, 10) (442,)
# Normal score : 0.46263830098374936
# CV score: [0.53864643 0.43212505 0.51425541 0.53344142 0.33325728]
# CV score mean : 0.47034511859358796
#  -------------------------------------------------------------------------------------
# shape - (442, 65)
# PolynomialFeatures score : 0.3992838684738401
# PolynomialFeatures CV score : [0.48845419 0.29536084 0.28899779 0.14002766 0.12733383]
# PolynomialFeatures CV score mean : 0.26803486278902
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
import numpy as np
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, stratify=y)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)



#2. 모델구성
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model.fit(x_train, y_train)



#3 훈련, 결과, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

print('Normal score :', model.score(x_test, y_test)) 
print('CV score:', scores)
print('CV score mean :', np.mean(scores))

# Normal score : 0.9333333333333333
# CV score: [0.91666667 1.         0.95833333 1.         0.91666667]
# CV score mean : 0.9583333333333334





# PolynomialFeatures -------------------------------------------------------------------------------------------------
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias=False 데이터를 증폭 시킬때 첫번째 열에 1을 안 나오게 한다 
xp = pf.fit_transform(x)                              # 결과에 따라 쓰는 정도의 파라미터로 사용
print(xp.shape)

x_train, x_test, y_train, y_test = train_test_split(xp, y, train_size=0.8, random_state=1234, stratify=y)



#2. 모델구성
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model.fit(x_train, y_train)



#3 훈련, 결과, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

print('PolynomialFeatures score :', model.score(x_test, y_test))
print('PolynomialFeatures CV score :', scores)
print('PolynomialFeatures CV score mean :', np.mean(scores))

# shape - (150, 4) (150,)
# Normal score : 0.9333333333333333
# CV score: [0.91666667 1.         0.95833333 1.         0.91666667]
# CV score mean : 0.9583333333333334
# -------------------------------------------------------------------------------------
# shape - (150, 14)
# PolynomialFeatures score : 0.9333333333333333
# PolynomialFeatures CV score : [0.91666667 1.         0.95833333 1.         0.91666667]
# PolynomialFeatures CV score mean : 0.9583333333333334
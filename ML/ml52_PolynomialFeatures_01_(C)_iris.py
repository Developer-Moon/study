from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression # 회귀, 이진분류
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, stratify=y)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)



#2. 모델구성
model = make_pipeline(StandardScaler(), LinearSVC())
model.fit(x_train, y_train)



#3 훈련, 결과, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
print('일반 score :', model.score(x_test, y_test)) 
print('CV :', scores)
print('CV 엔빵 :', np.mean(scores))








# PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False) # include_bias=False 데이터를 증폭 시킬때 첫번째에 1을 안 나오게 한다 
xp = pf.fit_transform(x)                              # 결과에 따라 쓰는 정도의 파라미터로 사용
print(xp.shape) # (150, 14)

x_train, x_test, y_train, y_test = train_test_split(xp, y, train_size=0.8, random_state=1234, stratify=y)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

#2. 모델구성
model = make_pipeline(StandardScaler(), LinearSVC())
model.fit(x_train, y_train)
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')

print('폴리 score :', model.score(x_test, y_test))
print('폴리CV :', scores)
print('폴리CV 엔빵 :', np.mean(scores))

# (150, 4) (150,)
# 일반 score : 0.8333333333333334
# CV : [0.83333333 0.95833333 0.91666667 1.         0.95833333]
# CV 엔빵 : 0.9333333333333333
# (150, 14)
# 폴리 score : 0.9333333333333333
# 폴리CV : [0.91666667 1.         0.95833333 1.         0.91666667]
# 폴리CV 엔빵 : 0.9583333333333334
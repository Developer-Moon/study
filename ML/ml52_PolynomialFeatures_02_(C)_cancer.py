from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer
import numpy as np
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
datasets = load_breast_cancer()
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

# shape - (569, 30) (569,)
# Normal score : 0.9736842105263158
# CV score: [0.94505495 0.93406593 0.97802198 0.96703297 0.96703297]
# CV score mean : 0.9582417582417582
#  -------------------------------------------------------------------------------------
# shape - (569, 495)
# PolynomialFeatures score : 0.9736842105263158
# PolynomialFeatures CV score : [0.96703297 0.98901099 0.95604396 0.97802198 0.97802198]
# PolynomialFeatures CV score mean : 0.9736263736263737
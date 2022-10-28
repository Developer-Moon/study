from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv',index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

train_set = train_set.fillna(0)

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)



#2. 모델구성
model = make_pipeline(MinMaxScaler(), LinearRegression())
model.fit(x_train, y_train)



#3 훈련, 결과, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold, scoring='r2')

print(x.shape)
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

# shape - (1459, 9)
# Normal score : 0.6030858858241663
# CV score: [0.51209011 0.60752332 0.57490351 0.60603941 0.64727029]
# CV score mean : 0.5895653278448545
#  -------------------------------------------------------------------------------------
# shape - (1459, 54)
# PolynomialFeatures score : 0.6392200393109853
# PolynomialFeatures CV score : [0.53690843 0.65076136 0.57241629 0.64373833 0.70400826]
# PolynomialFeatures CV score mean : 0.6215665357974975
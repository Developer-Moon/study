from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression # 회귀, 이진분류
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import QuantileTransformer, PowerTransformer 


# 데이터가 크면 로그변환 해도 좋을때가 있다

#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
model = LinearRegression()
# model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print('결과 :', round(result, 4))

# 결과 : 0.7665
# 결과 : 0.9145







# 로그변환
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df) # [506 rows x 13 columns]

df.plot.box()
plt.title('boston')
plt.xlabel('feature')
plt.ylabel('데이터값')
plt.show()

# print(df['B'].head())
# df['B'] = np.log1p(df['B']) # 로그변환
# print(df['B'].head())

df['CRIM'] = np.log1p(df['CRIM'])
df['ZN'] = np.log1p(df['ZN'])
df['TAX'] = np.log1p(df['TAX'])


x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=1234)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)



#2. 모델구성
model = LinearRegression()
# model = RandomForestRegressor()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print('로그변환 결과 :', round(result, 4))

# 결과 : 0.7665
# 결과 : 0.9145

# 로그변환 후 
# 결과 : 0.7711  
# 결과 : 0.9159



# 로그변환 결과 : 0.7596 - CRIM
# 로그변환 결과 : 0.7734 - ZN
# 로그변환 결과 : 0.7669 - TAX

# 로그변환 결과 : 0.7667 - CRIM, ZN, TAX 
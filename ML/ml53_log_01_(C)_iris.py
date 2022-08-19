from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression # 회귀, 이진분류
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다



#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
# model = LinearSVC()
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print('결과 :', round(result, 4))

# LR - 결과 : 1.0
# RF - 결과 : 1.0







# 로그변환
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df) # [506 rows x 13 columns]

df.plot.box()
plt.title('boston')
plt.xlabel('feature')
plt.ylabel('데이터값')
plt.show()

# print(df['B'].head())
df['sepal width'] = np.log1p(df['sepal width']) # 로그변환
# print(df['B'].head())



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

# LR - 결과 : 1.0
# RF - 결과 : 1.0

# 의미없다
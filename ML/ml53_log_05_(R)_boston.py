from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
# model = RandomForestRegressor()
model = LinearRegression()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print('Normal result :', round(result, 4))





# 로그변환 -------------------------------------------------------------------------------------------------
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
# print(df.info())
# df.plot.box()
# plt.title('boston')
# plt.xlabel('feature')
# plt.ylabel('데이터값')
# plt.show()

# 로그변환 log 1p 를 하는이유 log0 이 에러가 뜨기때문에 1을 더해주는것
# df['CRIM'] = np.log1p(df['CRIM']) 
df['ZN'] = np.log1p(df['ZN']) 
df['B'] = np.log1p(df['B']) 


x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=1234)


#2. 모델구성
# model = RandomForestRegressor()
model = LinearRegression()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print('log result :', round(result, 4))

# RF_Normal result : 0.9157
# RF_log result : 0.0.9145 - CRIM

# RF_Normal result : 0.9174
# RF_log result : 0.9162 -ZN

# RF_Normal result : 0.9119
# RF_log result : 0.9137 - B

# RF_Normal result : 0.918
# RF_log result : 0.9194 - All

#  ------------------------------------

# LR_Normal result : 0.7665
# LR_log result : 0.7596 - CRIM

# LR_Normal result : 0.7665
# LR_log result : 0.7734 - ZN

# LR_Normal result : 0.7665
# LR_log result : 0.7711 - B

# LR_Normal result : 0.7665
# LR_log result : 0.7779 - ZN, B [GOOD]

# LR_Normal result : 0.7665
# LR_log result : 0.7713 - All
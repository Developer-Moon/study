from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')



#1. 데이터
datasets = load_diabetes()
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
print('LR_Normal result :', round(result, 4))





# 로그변환 -------------------------------------------------------------------------------------------------
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
# print(df.info())
# df.plot.box()
# plt.title('boston')
# plt.xlabel('feature')
# plt.ylabel('데이터값')
# plt.show()

# 로그변환 log 1p 를 하는이유 log0 이 에러가 뜨기때문에 1을 더해주는것
# df['bmi'] = np.log1p(df['bmi']) 
# df['s1'] = np.log1p(df['s1']) 
df['s2'] = np.log1p(df['s2']) 
# df['s3'] = np.log1p(df['s3']) 
df['s4'] = np.log1p(df['s4']) 
# df['s5'] = np.log1p(df['s5']) 
df['s6'] = np.log1p(df['s6']) 



x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=1234)


#2. 모델구성
# model = RandomForestRegressor()
model = LinearRegression()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
result = r2_score(y_test, y_predict)
print('LR_log result :', round(result, 4), '- s6')

# RF_Normal result : 0.3864
# RF_log result : 0.4272 - bmi*

# RF_Normal result : 0.4086
# RF_log result : 0.3681 - s1

# RF_Normal result : 0.4116
# RF_log result : 0.4173 - s2*

# RF_Normal result : 0.41
# RF_log result : 0.4319 - s3*

# RF_Normal result : 0.4032
# RF_log result : 0.438 - s4*

# RF_Normal result : 0.438
# RF_log result : 0.4123 - s5

# RF_Normal result : 0.4001
# RF_log result : 0.4157 - s6*

# RF_Normal result : 0.3953
# RF_log result : 0.3973 - All[bmi, s2, s3, s4, s6]

#  ------------------------------------------------

# LR_Normal result : 0.4626
# LR_log result : 0.4603 - bmi

# LR_Normal result : 0.4626
# LR_log result : 0.4579 - s1

# LR_Normal result : 0.4626
# LR_log result : 0.4662 - s2*

# LR_Normal result : 0.4626
# LR_log result : 0.4624 - s3

# LR_Normal result : 0.4626
# LR_log result : 0.464 - s4*

# LR_Normal result : 0.4626
# LR_log result : 0.4612 - s5

# LR_Normal result : 0.4626
# LR_log result : 0.4627 - s6* 

# LR_Normal result : 0.4626
# LR_log result : 0.4674 - All [s2, s4, s6]
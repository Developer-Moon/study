from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 데이터에따라 로그변환이 스케일러보다 좋을때가 있다

#1. 데이터
datasets = load_iris()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234, stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
result = accuracy_score(y_test, y_predict)
print('Normal result :', round(result, 4))





# 로그변환 -------------------------------------------------------------------------------------------------
df = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
df.plot.box()
plt.title('boston')
plt.xlabel('feature')
plt.ylabel('데이터값')
plt.show()

# 로그변환 log 1p 를 하는이유 log0 이 에러가 뜨기때문에 1을 더해주는것
df['sepal width (cm)'] = np.log1p(df['sepal width (cm)']) 

x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=1234, stratify=y)


#2. 모델구성
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)
result = accuracy_score(y_test, y_predict)
print('log result :', round(result, 4))

# Normal result : 0.9333
# log result : 0.9333
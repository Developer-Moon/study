from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, VotingRegressor 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings(action='ignore')


#1. 데이터 
datasets = load_boston()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
print(df.head(7)) # 7개 나온다

x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=123)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
xg = XGBRegressor()
lg = LGBMRegressor(verbose=0)
cat = CatBoostRegressor() # action='ignore'

model = VotingRegressor(estimators=[('XG', xg), ('LG', lg), ('CAT', cat)])



#3. 훈련
model.fit(x_train, y_train)



#4. 평가, 예측
y_predict = model.predict(x_test)

score = r2_score(y_test, y_predict)
print('보팅결과 :', round(score, 4)) # 보팅결과 : 0.9912

calssifiers =[xg, lg, cat]
for model2 in calssifiers:
    model2.fit(x_train, y_train)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__ # 모델의 이름을 반환
    print('{0} 정확도 : {1:.4f}'.format(class_name, score2))

# 보팅결과 : 0.9912
# XGBClassifier 정확도 : 0.9912
# LGBMClassifier 정확도 : 0.9912
# CatBoostClassifier 정확도 : 0.9912


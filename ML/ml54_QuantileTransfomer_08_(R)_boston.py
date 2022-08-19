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
from xgboost import XGBClassifier,XGBRegressor
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import QuantileTransformer, PowerTransformer # 이상치에 자유롭다



#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1234)

# PowerTransformer(method='yeo-johnson'), PowerTransformer(method='box-cox')
scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer(method='yeo-johnson'),
           ] # PowerTransformer(method='box-cox')
models = [LinearRegression(), RandomForestRegressor(), XGBRegressor(verbose=False)] # , XGBRegressor

for i in scalers:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for model in models:
        try:
            model = model
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
            result = r2_score(y_test, y_predict)    
            print(model, scaler,'- 결과 :', round(result, 4))    
        except:
            print(model, 'X')   
        




    # for model in models:
    #     if model == PowerTransformer:
    #         try:
    #             model = model()
    #             model.fit(x_train, y_train)
    #             y_predict = model.predict(x_test)
    #             result = r2_score(y_test, y_predict)    
    #             print(model, scaler,'결과 :', round(result, 4))    
    #         except:
    #             print(model, 'X')   
    #     else:    
    #         model = model()
    #         model.fit(x_train, y_train)
    #         y_predict = model.predict(x_test)
    #         result = r2_score(y_test, y_predict)    
    #         print(model, scaler,'결과 :', round(result, 4))    
                
                
        



import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import matplotlib.pyplot as plt

import xgboost as xgb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정



path = './_data/dacon_antena/'      
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

train = pd.read_csv(path + 'train.csv')

train_x = train.filter(regex='X') # Input : X Featrue
train_y = train.filter(regex='Y') # Output : Y Feature

xgb = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100,
                                            learning_rate=0.08,
                                            gamma = 0,
                                            subsample=0.75,
                                            colsample_bytree = 1,
                                            max_depth=7) ).fit(train_x, train_y)


test_x = pd.read_csv(path + 'test.csv').drop(columns=['ID'])

preds = xgb.predict(test_x)
print(xgb.score(train_x, train_y))

submit = pd.read_csv(path +'sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]

submit.to_csv(path + 'submission_m.csv', index=False)




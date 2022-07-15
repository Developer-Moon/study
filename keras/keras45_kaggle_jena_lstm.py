# 평가까지
# lstm Conv1D  
# 스플릿트 사용(컬럼이 뭉치로 사용되어야 한다)

import numpy as np
from sklearn import metrics          
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D 
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score, mean_squared_error  
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

  
train_set = pd.read_csv('./_data/kaggle_jena/jena_climate_2009_2016.csv')


size = 5
def split_x(seq, size):  
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)
train = split_x(train_set, size)

print('------------------')
print(train)
print(train.shape)
 
x = train[:, :-1]
y = train[:, -1] 

print(x.shape)
print(y.shape) # (420547, 15)



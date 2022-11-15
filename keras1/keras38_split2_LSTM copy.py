import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_breast_cancer  
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten, GRU, LSTM
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score, mean_squared_error  
from sklearn.metrics import r2_score 
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  



amore = np.array(range(100))                        
samsung = np.array(range(101, 201)) 


size = 5

def split_x(seq, size):  
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)



x1 = split_x(amore, size)
x2 = split_x(samsung, size)
y = split_x(amore, size)


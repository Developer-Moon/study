import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_breast_cancer  
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten, GRU
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score, mean_squared_error  
from sklearn.metrics import r2_score 
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

 
trainset = np.array(range(1,101))
testset = np.array(range(96,106))
print(len(trainset))

size = 5
def split_x(seq, size):  
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)
train = split_x(trainset, size)

print('------------------')
print(train)
print(train.shape)
 
x = train[:, :-1]
y = train[:, -1] 
 
print(x.shape) 
print(y.shape) 




size = 4
test = split_x(testset, size)

print(test)
print(x.shape, y.shape, test.shape)



model = Sequential()
model.add(Dense(33, activation = 'relu', input_shape=(4,)))
model.add(Dense(5))
model.add(Dense(1))
# LSTM을 DNN으로 구현 가능
 
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1, verbose=1)



y_pred = model.predict(test)
print(y_pred)


# [[ 99.996666]
#  [100.9965  ]
#  [101.99633 ]
#  [102.99616 ]
#  [103.996   ]
#  [104.995834]
#  [105.99566 ]]








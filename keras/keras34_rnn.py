import numpy as np      
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense, SimpleRNN

#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  
# y = ?
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8,], [7, 8, 9]])
x = np.array([4, 5, 6, 7, 8, 9, 10])

# RNN은 shape가 3차원 - (N, 3, 1) 여기서 3은 자르는 단위 

# input_shape = (행, 열, 몇개씩 짜르는지!!!) 





from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
import numpy as np  

# train_test_split으로만 10 : 3 : 3 으로 나눠라 

#1. 데이터 
x = np.array(range(1, 17))
y = np.array(range(1, 17))


#train, test 6:4로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.65, random_state = 42)
print(x_train) # [ 9 10  3 16  5  8 11 13  4  7]
print(x_test)  # [ 1  2  6 15 14 12]


#test set을 5:5로 test, val 나누기
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state = 42)
print(x_train) # [ 9 10  3 16  5  8 11 13  4  7]
print(x_test)  # [ 6 14 15]
print(x_val)   # [ 1  2 12]
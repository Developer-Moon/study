from tensorflow.python.keras.models import Sequential                              # 컬러 - softmax 100개
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split        
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd 


import tensorflow as tf            
tf.random.set_seed(66)         #텐서플로의 난수표 66번 사용 이 난수는 w = 웨이트

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  
print(x_test.shape, y_test.shape)  


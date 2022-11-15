from keras.datasets import imdb
import numpy as np                             
import pandas as pd             

(x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=10000
)

print(x_train)
print(y_train)
print(x_train.shape, x_test.shape) # (25000,) (25000,)

print(np.unique(y_train, return_counts=True)) # 2개의 뉴스카테고리
print(len(np.unique(y_train)))                # 2개           

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train[0]))             # <class 'list'> 
# print(x_train[0].shape)           # error : 'list' object has no attribute 'shape' - list는 shape을 취급하지 않는다
print(len(x_train[0])) # 218 리스트의 길이가 다 다르다
print(len(x_train[1])) # 189 리스트의 길이가 다 다르다


# 8982중 가장 높은값을 찾기 위해선 for문을 써야 한다

max(len(i) for i in x_train) # x_train반복하는 값들이 len(i)안에 들어간다 그걸 max값으로

print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) # 2494
print("뉴스기사의 평균길이 : ", sum(map(len, x_train)) / len(x_train)) #  238.71364

# 전처리
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

print(x_train)

x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre') # 앞에서부터 0으로 채우고 100개까지
                      # (8982,) -> (8982, 100)
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre') # 앞에서부터 0으로 채우고 100개까지


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape) # (8982, 100) (8982, 46)
print(x_test.shape, y_test.shape)   # (2246, 100) (2246, 46)



#2. 모델구성
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding, Conv1D

model = Sequential()
model.add()


 
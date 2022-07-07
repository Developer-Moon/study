import numpy as np 
from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense    

#1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],                   
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],           
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])            
y = np.array([11,12,13,14,15,16,17,18,19,20])   


print(x.shape, y.shape) # (3, 10)   (10, )
x = x.T 
print(x.shape)          # (10, 3)


#2. 모델구성
model = Sequential()
# model.add(Dense(10, input_dim=3))     # (10,3)ㅣ
model.add(Dense(10, input_shape=(3, ))) # 컬럼이 3개   열을 제외한 나머지가 들어감  2차원이상은 shape
model.add(Dense(5)) 
model.add(Dense(3))
model.add(Dense(1))
model.summary()

"""
Model: "sequential"

Layer (type)                 Output Shape              Param #        input_dim=3
=================================================================
dense (Dense)                (None, 10)                40             None, 는 행의 개수
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 4
=================================================================
Total params: 117
Trainable params: 117
Non-trainable params: 0



Layer (type)                 Output Shape              Param #       input_shape=(3, )
=================================================================
dense (Dense)                (None, 10)                40
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 4
=================================================================
Total params: 117
Trainable params: 117
Non-trainable params: 0
"""



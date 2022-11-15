import numpy as np


import tensorflow as tf            
tf.random.set_seed(66)


#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)])                         # 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)])   # 원유, 돈육, 밀
x3_datasets = np.array([range(100, 200), range(1301, 1401)])                  # 우리반 아이큐, 우리반 키
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape ) # (100, 2) (100, 3) (100, 2)

y1 = np.array(range(2001, 2101)) # 금리
y2 = np.array(range(201, 301))  

print(y1.shape) # (100,)
print(y2.shape) # (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test,y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, x3, y1, y2,
     train_size=0.75, random_state=66
 )

print(x1_train.shape, x1_test.shape) # (75, 2) (25, 2)
print(x2_train.shape, x2_test.shape) # (75, 3) (25, 3)
print(x3_train.shape, x3_test.shape) # (75, 2) (25, 2)
print(y1_train.shape, y1_test.shape) # (75,) (25,)
print(y2_train.shape, y2_test.shape) # (75,) (25,)



#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, SimpleRNN

#2-1. 모델1
input_01 = Input(shape=(2,))
dense_01 = Dense(100, activation='relu', name='love_ys1')(input_01)
dense_02 = Dense(100, activation='relu', name='love_ys2')(dense_01)
dense_03 = Dense(100, activation='relu', name='love_ys3')(dense_02)
output_01 = Dense(100, activation='relu', name='out_ys1')(dense_03)

#2-2 모델2
input_02 = Input(shape=(3,))
dense_11 = Dense(100, activation='relu', name='love_ys11')(input_02)
dense_12 = Dense(100, activation='relu', name='love_ys12')(dense_11)
dense_13 = Dense(100, activation='relu', name='love_ys13')(dense_12) 
dense_14 = Dense(100, activation='relu', name='love_ys14')(dense_13)
output_02 = Dense(100, activation='relu', name='out_ys2')(dense_14)

#2-3 모델3
input_03 = Input(shape=(2,))
dense_001 = Dense(100, activation='relu', name='love_ys111')(input_03)
dense_002 = Dense(100, activation='relu', name='love_ys112')(dense_001)
dense_003 = Dense(100, activation='relu', name='love_ys113')(dense_002)
dense_004 = Dense(100, activation='relu', name='love_ys114')(dense_003)
dense_005 = Dense(100, activation='relu', name='love_ys115')(dense_004)
output_03 = Dense(100, activation='relu', name='out_ys3')(dense_005)

# Concatenate
from tensorflow.python.keras.layers import Concatenate, concatenate
merge_01 = concatenate([output_01, output_02, output_03]) # concatenate 사슬처럼 엮다 list의 append개념             모델1과 모델2가 합쳐져서 43개의 output이 나온다
merge_02 = Dense(100, activation='relu', name='mg_01')(merge_01)
merge_03 = Dense(100, name='mg_02')(merge_02)
last_output_01 = Dense(1)(merge_03)                       # y값이 2개일때 아웃풋을 2개로 나눈다
last_output_02 = Dense(1)(merge_03)

#2-4. output모델1



model = Model(inputs=[input_01, input_02, input_03], outputs=[last_output_01, last_output_02]) # intput이 2개 이상이라 list
model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train], epochs=200, batch_size=10, validation_split=0.2)



#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])
print('loss : ', loss) 

y1_predict, y2_predict = model.predict([x1_test, x2_test, x3_test])


from sklearn.metrics import r2_score 
y1_r2 = r2_score(y1_test, y1_predict)
y2_r2 = r2_score(y2_test, y2_predict)

print(y1_r2)
print(y2_r2)

#               합                    y1_predict           y2_predict
# loss : [0.015889838337898254, 0.01006096601486206, 0.0058288718573749065]
# y1_r2: 0.9999887344851828
# y2_r2: 0.9999934755246629
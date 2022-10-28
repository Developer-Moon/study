import numpy as np

import tensorflow as tf            
tf.random.set_seed(66)




# 만들수 있지만 안하고 안나온다 


#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)])                         # 삼성전자 종가, 하이닉스 종가

x1 = np.transpose(x1_datasets)

print(x1.shape) # (100, 2)

y1 = np.array(range(2001, 2101)) # 금리
y2 = np.array(range(201, 301))  

print(y1.shape) # (100,)
print(y2.shape) # (100,)

from sklearn.model_selection import train_test_split
x1_train, x1_test ,y1_train, y1_test, y2_train, y2_test = train_test_split(x1, y1, y2,
     train_size=0.75, random_state=66
 )

print(x1_train.shape, x1_test.shape) # (75, 2) (25, 2)
print(y1_train.shape, y1_test.shape) # (75,) (25,)
print(y2_train.shape, y2_test.shape) # (75,) (25,)



#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, SimpleRNN

#2-1. 모델
input_01 = Input(shape=(2,))
dense_01 = Dense(100, activation='relu', name='love_ys1')(input_01)
dense_02 = Dense(100, activation='relu', name='love_ys2')(dense_01)
dense_03 = Dense(100, activation='relu', name='love_ys3')(dense_02)
output_01 = Dense(100, activation='relu', name='out_ys1')(dense_03)


# Concatenate
from tensorflow.python.keras.layers import Concatenate, concatenate
# merge_01 = concatenate([output_01, output_02, output_03]) # concatenate 사슬처럼 엮다 list의 append개념             모델1과 모델2가 합쳐져서 43개의 output이 나온다
# merge_01 = Concatenate(axis=1)([output_01, output_02, output_03])
merge_02 = Dense(100, activation='relu', name='mg_01')(output_01)
merge_03 = Dense(100, name='mg_02')(merge_02)
last_output = Dense(1)(merge_03)                       # y값이 2개일때 아웃풋을 2개로 나눈다


#2-4. output모델1
last_output41 = Dense(1)(last_output)
last_output42 = Dense(1)(last_output41)
last_output01 = Dense(1)(last_output42)


#2-4. output모델2
last_output51 = Dense(1)(last_output)
last_output52 = Dense(1)(last_output51)
last_output02 = Dense(1)(last_output52)


model = Model(inputs=input_01, outputs=[last_output01, last_output02]) # intput이 2개 이상이라 list
model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x1_train, [y1_train, y2_train], epochs=300, batch_size=10, validation_split=0.2)



#4. 평가, 예측
loss = model.evaluate(x1_test, [y1_test, y2_test])
print('loss : ', loss) 

y1_predict, y2_predict = model.predict(x1_test)


from sklearn.metrics import r2_score 
y1_r2 = r2_score(y1_test, y1_predict)
y2_r2 = r2_score(y2_test, y2_predict)

print(y1_r2)
print(y2_r2)

#               합               y1_predict           y2_predict
# loss :  [717.8025512695312, 9.630058288574219, 708.1724853515625]
# 0.9892207143143529
# 0.20731599774165688
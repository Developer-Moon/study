import numpy as np

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)])                         # ex) 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)])   # ex) 원유, 돈육, 밀
x3_datasets = np.array([range(100, 200), range(1301, 1401)])                  # ex) 우리반 아이큐, 우리반 키
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)
x3 = np.transpose(x3_datasets)

print(x1.shape, x2.shape, x3.shape ) # (100, 2) (100, 3) (100, 2)

y = np.array(range(2001, 2101)) # 금리

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1, x2, x3, y,
     train_size=0.75, random_state=66
 )

print(x1_train.shape, x1_test.shape) # (75, 2) (25, 2)
print(x2_train.shape, x2_test.shape) # (75, 3) (25, 3)
print(x3_train.shape, x3_test.shape) # (75, 2) (25, 2)
print(y_train.shape, y_test.shape)   # (75,) (25,)



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


from tensorflow.python.keras.layers import Concatenate, concatenate
merge_01 = concatenate([output_01, output_02, output_03]) # concatenate 사슬처럼 엮다 list의 append개념             모델1과 모델2가 합쳐져서 43개의 output이 나온다
merge_02 = Dense(100, activation='relu', name='mg_01')(merge_01)
merge_03 = Dense(100, name='mg_02')(merge_02)
last_output = Dense(1)(merge_03)

model = Model(inputs=[input_01, input_02, input_03], outputs=last_output) # intput이 2개 이상이라 list
model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], y_train, epochs=200, batch_size=10, validation_split=0.2)



#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], y_test)
print('loss : ', loss) 

y_predict = model.predict([x1_test, x2_test, x3_test])
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# loss :  0.09184157103300095
# r2스코어 : 0.9998972012505007


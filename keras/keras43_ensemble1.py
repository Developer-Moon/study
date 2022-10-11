import numpy as np

# 앙상블 학습(Ensemble Learning)은 여러 개의 분류기를 생성하고, 그 예측을 결합함으로써 보다 정확한 예측을 도출하는 기법

#1. 데이터
x1_datasets = np.array([range(100), range(301, 401)])                         # ex) 삼성전자 종가, 하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)])   # ex) 원유, 돈육, 밀
x1 = np.transpose(x1_datasets)
x2 = np.transpose(x2_datasets)

print(x1.shape, x2.shape)       # (100, 2) (100, 3)

y = np.array(range(2001, 2101)) # ex) 금리

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,
     train_size=0.75, random_state=66
 )

print(x1_train.shape, x1_test.shape)   # (75, 2) (25, 2)
print(x2_train.shape, x2_test.shape)   # (75, 3) (25, 3)
print(y_train.shape, y_test.shape)     # (75,) (25,)



#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, SimpleRNN

#2-1. 모델1
input_01 = Input(shape=(2,))
dense_01 = Dense(100, activation='relu', name='love_ys1')(input_01)
# dropout_01 = Dropout(0.5)(dense_01)
dense_02 = Dense(100, activation='relu', name='love_ys2')(dense_01)
dense_03 = Dense(100, activation='relu', name='love_ys3')(dense_02)
output_01 = Dense(100, activation='relu', name='out_ys1')(dense_03)

#2-2 모델2
input_02 = Input(shape=(3,))
dense_11 = Dense(100, activation='relu', name='love_ys11')(input_02)
# dropout_11 = Dropout(0.5)(dense_11)
dense_12 = Dense(100, activation='relu', name='love_ys12')(dense_11)
dense_13 = Dense(100, activation='relu', name='love_ys13')(dense_12) 
dense_14 = Dense(100, activation='relu', name='love_ys14')(dense_13)
output_02 = Dense(100, activation='relu', name='out_ys2')(dense_14)

from tensorflow.python.keras.layers import Concatenate, concatenate   # concatenate 사슬처럼 엮다 list의 append개념
merge_01 = concatenate([output_01, output_02])                        # 모델1과 모델2가 합쳐져서 200개의 output이 나온다
merge_02 = Dense(100, activation='relu', name='mg_01')(merge_01)
merge_03 = Dense(100, name='mg_02')(merge_02)
last_output = Dense(1)(merge_03)

model = Model(inputs=[input_01, input_02], outputs=last_output)       # intput이 2개 이상이라 list
model.summary()



#3. 컴파일, 훈련#
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=200, batch_size=10, validation_split=0.2)



#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)

print('loss : ', loss) 

y_predict = model.predict([x1_test, x2_test])
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# loss :  0.004217991139739752
# r2스코어 : 0.9999952818006725


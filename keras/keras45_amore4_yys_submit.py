import numpy as np
from sklearn import datasets
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import time
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. 데이터
path = './_data/test_amore_0718/'
dataset_sam = pd.read_csv(path + '삼성전자220718.csv', thousands=',', encoding='cp949')
dataset_amo = pd.read_csv(path + '아모레220718.csv', thousands=',', encoding='cp949')

dataset_sam = dataset_sam.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)
dataset_amo = dataset_amo.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)

dataset_amo = dataset_amo.rename(columns={'Unnamed: 6':'증감량'})
dataset_sam = dataset_sam.rename(columns={'Unnamed: 6':'증감량'})

dataset_sam = dataset_sam.fillna(0)
dataset_amo = dataset_amo.fillna(0)

dataset_sam = dataset_sam.loc[dataset_sam['일자']>="2018/05/04"] # 액면분할 이후 데이터만 사용
dataset_amo = dataset_amo.loc[dataset_amo['일자']>="2018/05/04"] # 삼성의 액면분할 날짜 이후의 행개수에 맞춰줌
print(dataset_amo.shape, dataset_sam.shape) # (1035, 11) (1035, 11)

dataset_sam = dataset_sam.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순 정렬
dataset_amo = dataset_amo.sort_values(by=['일자'], axis=0, ascending=True)
#print(dataset_amo.head) # 앞 다섯개만 보기


dataset_sam = dataset_sam.drop(['일자'], axis=1)
dataset_amo = dataset_amo.drop(['일자'], axis=1)
dataset_sam = dataset_sam[['시가', '고가', '저가', '증감량', '등락률', '거래량', '기관', '외국계', '종가']]
dataset_amo = dataset_amo[['시가', '고가', '저가', '증감량', '등락률', '거래량', '기관', '외국계', '종가']]

print(dataset_amo)
print(dataset_sam)



dataset_sam = np.array(dataset_sam) 
dataset_amo = np.array(dataset_amo)



print(dataset_amo)


print(dataset_sam)


# 시계열 데이터 만드는 함수

time_steps = 20
y_column = 3

def split_xy(dataset, time_steps, y_column):                 
    x = []
    y = []
    for i in range(len(dataset)):
        x_end_number = i + time_steps      # 0 + 5   > 5
        y_end_number = x_end_number + y_column - 1    # 5 + 3 -1 > 7
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, : -1]                   # 0 : 5 , : -1   > 0행~4행, 마지막열 뺀 전부
        tmp_y = dataset[x_end_number-1:y_end_number, -1]       # 5 - 1 : 7 , -1  > 마지막 열의 4~6행
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy(dataset_amo, time_steps, y_column)
x2, y2 = split_xy(dataset_sam, time_steps, y_column)

print(x1.shape) # (1014, 20, 8)
print(x2.shape) # (1014, 20, 8)
print(y1.shape) # (1029, 5)
print(y2.shape) # (1029, 5)


x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y_2test = train_test_split(x1, x1, y1, y2, test_size=0.2, shuffle=False)


print(x1_train.shape, x1_test.shape) # (811, 20, 8) (203, 20, 8)
print(x2_train.shape, x2_test.shape) # (811, 20, 8) (203, 20, 8)






# data 스케일링
scaler = MinMaxScaler()

x1_train = x1_train.reshape(811*20, 8)
x1_train = scaler.fit_transform(x1_train)
x1_test = x1_test.reshape(203*20, 8)
x1_test = scaler.transform(x1_test)

x2_train = x2_train.reshape(811*20, 8)
x2_train = scaler.fit_transform(x2_train)
x2_test = x2_test.reshape(203*20, 8)
x2_test = scaler.transform(x2_test)

# Conv1D에 넣기 위해 3차원화
x1_train = x1_train.reshape(811, 20, 8)
x1_test = x1_test.reshape(203, 20, 8)
x2_train = x2_train.reshape(811, 20, 8)
x2_test = x2_test.reshape(203, 20, 8)

# # 2. 모델구성
# # 2-1. 모델1
# input1 = Input(shape=(20, 8))
# dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
# dense2 = LSTM(128, activation='relu', name='d2')(dense1)
# dense3 = Dense(64, activation='relu', name='d3')(dense2)
# output1 = Dense(32, activation='relu', name='out_d1')(dense3)

# # 2-2. 모델2
# input2 = Input(shape=(20, 8))
# dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
# dense12 = LSTM(128, activation='swish', name='d12')(dense11)
# dense13 = Dense(64, activation='relu', name='d13')(dense12)
# dense14 = Dense(32, activation='relu', name='d14')(dense13)
# output2 = Dense(16, activation='relu', name='out_d2')(dense14)

# from tensorflow.python.keras.layers import concatenate
# merge1 = concatenate([output1, output2], name='m1')
# merge2 = Dense(100, activation='relu', name='mg2')(merge1)
# merge3 = Dense(100, name='mg3')(merge2)
# last_output = Dense(1, name='last')(merge3)

# model = Model(inputs=[input1, input2], outputs=[last_output])

model = load_model('./_test/keras46_siga3.h5')

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
start_time = time.time()
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
fit_log = model.fit([x1_train, x2_train], y1_train, epochs=300, batch_size=16, callbacks=[Es], validation_split=0.1)
end_time = time.time()
model.save('./_save/keras46_siga3.h5')

# 4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y1_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('prdict: ', predict[-1]) # 제일 마지막에 나온거 하나 슬라이싱
print('걸린 시간: ', end_time-start_time)



# loss:  21554392.0
# prdict:  [134802.34]
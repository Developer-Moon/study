from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, SimpleRNN, LSTM, Conv1D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
import numpy as np                                               
import pandas as pd



#1. 데이터
amore = pd.read_csv('./_data/test_amore_0718/아모레220718.csv', encoding = 'CP949', thousands=',')
samsung = pd.read_csv('./_data/test_amore_0718/삼성전자220718.csv', encoding = 'CP949', thousands=',')
# ['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']

print(amore.info())

amore = amore.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)      
samsung = samsung.drop(['전일비','금액(백만)','신용비','개인','외인(수량)','프로그램','외인비'], axis=1)

amore = amore.rename(columns={'Unnamed: 6':'증감량'})       # rename
samsung = samsung.rename(columns={'Unnamed: 6':'증감량'})   # rename

amore = amore.fillna(0)
samsung = samsung.fillna(0)

amore = amore.loc[amore['일자']>="2018/05/04"] # 액면분할 이후 데이터만 사용
samsung = samsung.loc[samsung['일자']>="2018/05/04"] # 삼성의 액면분할 날짜 이후의 행개수에 맞춰줌
print(amore.shape, samsung.shape) # (1035, 10) (1035, 10)

amore = amore.sort_values(by=['일자'], axis=0, ascending=True)      # 오름차순 정렬
samsung = samsung.sort_values(by=['일자'], axis=0, ascending=True)    

feature_cols = ['시가', '고가', '저가', '종가', '증감량', '등락률', '거래량'] # cols로 원하는 columns들 지정 [이 좋은걸 내가 몰랐네;;]
label_cols = ['시가']                                                      # 내가 원하는 값 지정 


size = 20                                      
def split_x(dataset, size):                   
    a1 = []                                
    for i in range(len(dataset) - size + 1):   
        subset = dataset[i : (i + size)]      
        a1.append(subset)         

    return np.array(a1)                      

x1 = split_x(amore[feature_cols], size)
x2 = split_x(samsung[feature_cols], size)
y = split_x(amore[label_cols], size)       # amore의 시가

print(amore.info())


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,
    train_size=0.8, shuffle=False
    )

print(x1_train.shape, x1_test.shape) # (812, 20, 7) (204, 20, 7)
print(x2_train.shape, x2_test.shape) # (812, 20, 7) (204, 20, 7)

scaler = MinMaxScaler()
x1_train, x2_train = x1_train.reshape(812*20, 7), x2_train.reshape(812*20, 7)
x1_test, x2_test = x1_test.reshape(204*20, 7), x2_test.reshape(204*20, 7)

x1_train, x2_train = scaler.fit_transform(x1_train), scaler.fit_transform(x2_train)
x1_test, x2_test = scaler.transform(x1_test), scaler.transform(x2_test)

x1_train, x2_train = x1_train.reshape(812, 20, 7), x2_train.reshape(812, 20, 7)
x1_test, x2_test = x1_test.reshape(204, 20, 7), x2_test.reshape(204, 20, 7)

print(x1_train.shape, x1_test.shape) # (775, 2, 9) (259, 2, 9)
print(x2_train.shape, x2_test.shape) # (775, 2, 9) (259, 2, 9)


#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, SimpleRNN


#2-1. model : amore
input1 = Input(shape=(20, 7))
dense1 = Conv1D(64, 2, activation='relu', name='d1')(input1)
dense2 = LSTM(64, activation='relu', name='d2')(dense1)
dense3 = Dense(64, activation='relu', name='d3')(dense2)
output1 = Dense(32, activation='relu', name='out_d1')(dense3)

# 2-2. 모델2
input2 = Input(shape=(20, 7))
dense11 = Conv1D(64, 2, activation='relu', name='d11')(input2)
dense12 = LSTM(64, activation='swish', name='d12')(dense11)
dense13 = Dense(64, activation='relu', name='d13')(dense12)
dense14 = Dense(32, activation='relu', name='d14')(dense13)
output2 = Dense(16, activation='relu', name='out_d2')(dense14)

from tensorflow.python.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='m1')
merge2 = Dense(100, activation='relu', name='mg2')(merge1)
merge3 = Dense(50, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=[last_output]) 
model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')             
################################
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint  
import datetime
date = datetime.datetime.now()    
print(date)                        # 2022-07-07 17:25:03.261773
date = date.strftime('%m%d_%H%M') 
print(date)                        # 0707_1724


filepath = './_ModelCheckpoint/k24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'   
#                   4번재 자리      소수 4번째 자리
################################
              
"""
mcp = ModelCheckpoint(moniter='val_loss', mode='auto', verbose=1,
                      save_best_only=True,                                          
                      filepath="".join([filepath, 'amore_', date, '_', filename])     
                    # filepath='./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5' 
                      )
""" 
model.compile(loss='mse', optimizer='adam')
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150, restore_best_weights=True)
hist = model.fit([x1_train, x2_train], y_train, epochs=500, batch_size=30, callbacks=[earlyStopping], validation_split=0.2)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
predict = model.predict([x1_test, x2_test])
print('loss: ', loss)
print('prdict: ', predict[-1:]) # 제일 마지막에 나온거 하나 슬라이싱




#  loss:  345680642048.0
# prdict:  [[508648.84]]
# 85 92 100 node 값 반으로        5만원정도 줄었어요 


# loss:  318392074240.0
# prdict:  [[479028.75]] 
# validation_split 0.1 -> 0.2  patience 100 -> 150        3만원정도 줄었어요

# 배치사이즈 64개에서 30개로 
# loss:  220421537792.0
# prdict:  [[443373.12]]    3 만원 정도 줄었어요


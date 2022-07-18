from tensorflow.python.keras.models import Sequential, Model, Input   
from tensorflow.python.keras.layers import Dense, Input, Dropout, SimpleRNN, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error   
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler    
import numpy as np                                               
import pandas as pd



#1. 데이터
amore = pd.read_csv('./_data/test_amore_0718/amore.csv', encoding = 'CP949', thousands=',')
samsung = pd.read_csv('./_data/test_amore_0718/samsung.csv', encoding = 'CP949', thousands=',')
# print(amore.columns)
# print(samsung.columns)
# ['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']

amore = amore.sort_values(by=['일자'], axis=0, ascending=True) # 오름차순 정렬
samsung = samsung.sort_values(by=['일자'], axis=0, ascending=True)    


amore['일자'] = pd.to_datetime(amore['Datetime'])
print(amore)



"""
amore = amore.dropna()
amore = amore.drop(['전일비', 'Unnamed: 6', '등락률', '개인', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
amore = amore[:1035] # 앙상블 사용시 행의 개수가 같아야 한다


samsung = samsung.dropna()
samsung = samsung.drop(['전일비', 'Unnamed: 6', '등락률', '개인', '외인(수량)', '외국계', '프로그램', '외인비'], axis=1)
samsung = samsung[:1035] # 앙상블 사용시 행의 개수가 같아야 한다


   
     
     
              
                            
size = 2                                      
def split_x1(dataset, size):                   
    a1 = []                                  
    for i in range(len(dataset) - size + 1):   
        subset = dataset[i : (i + size)]      
        a1.append(subset)                     
    return np.array(a1)                      

x1 = split_x1(amore, size) 
x2 = split_x1(samsung, size)     
y = split_x1(amore['시가'], size) 


x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,
    train_size=0.75, 
    )


print(x1_train.shape, x1_test.shape) # (775, 2, 9) (259, 2, 9)
print(x2_train.shape, x2_test.shape) # (775, 2, 9) (259, 2, 9)

x1_train = x1_train.reshape(775, 2*9)
x1_test = x1_test.reshape(259, 2*9)
x2_train = x2_train.reshape(775, 2*9)
x2_test = x2_test.reshape(259, 2*9)


scaler = MinMaxScaler()
# scaler = StandardScaler()  
# scaler = MaxAbsScaler()                                                                                  
# scaler = RobustScaler()
x1_train = scaler.fit_transform(x1_train) 
x2_train = scaler.fit_transform(x2_train) 
x1_test = scaler.transform(x1_test)
x2_test = scaler.transform(x2_test)


x1_train = x1_train.reshape(775, 2, 9)
x1_test = x1_test.reshape(259, 2, 9)
x2_train = x2_train.reshape(775, 2, 9)
x2_test = x2_test.reshape(259, 2, 9)



#2. 모델구성
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, SimpleRNN

#2-1. model : amore
input_01 = Input(shape=(2, 9))
amore_dense_01 = Conv1D(100, 2, activation='relu', name='amore_01')(input_01)
amore_dense_02 = LSTM(100, activation='relu', name='amore_02')(amore_dense_01)
amore_dense_03 = Dense(100, activation='relu', name='amore_03')(amore_dense_02)
amore_dense_04 = Dense(100, activation='relu', name='amore_04')(amore_dense_03)
amore_dense_05 = Dense(100, activation='relu', name='amore_05')(amore_dense_04)
output_01 = Dense(100, activation='relu', name='amore_06')(amore_dense_05)



#2-2 model : samsung
input_02 = Input(shape=(2, 9))
samsung_dens_01 = Conv1D(100, 2, activation='relu', name='samsung_01')(input_02)
samsung_dens_02 = LSTM(100, activation='relu', name='samsung_02')(samsung_dens_01)
samsung_dens_03 = Dense(100, activation='relu', name='samsung_03')(samsung_dens_02)
samsung_dens_04 = Dense(100, activation='relu', name='samsung_04')(samsung_dens_03)
samsung_dens_05 = Dense(100, activation='relu', name='samsung_05')(samsung_dens_04)
output_02 = Dense(100, activation='relu', name='samsung_06')(samsung_dens_05)


# Concatenate
from tensorflow.python.keras.layers import Concatenate, concatenate
merge_01 = concatenate([output_01, output_02])
merge_02 = Dense(100, activation='relu', name='mg_01')(merge_01)
merge_03 = Dense(100, name='mg_02')(merge_02)
last_output_01 = Dense(1)(merge_03)                      


model = Model(inputs=[input_01, input_02], outputs=[last_output_01]) 
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
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)                 
mcp = ModelCheckpoint(moniter='val_loss', mode='auto', verbose=1,
                      save_best_only=True,                                          
                      filepath="".join([filepath, 'amore_', date, '_', filename])     
                    # filepath='./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5' 
                      )

hist = model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=100, verbose=1, callbacks=[earlyStopping, mcp], validation_split=0.2)



#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss) 

y_predict = model.predict([x1_test, x2_test])

print(y_predict)

# from sklearn.metrics import r2_score 
# r2 = r2_score(y_test, y_predict)
# print('r2스코어 :', r2)

"""

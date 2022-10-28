from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score 
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  



#1. 데이터
datasets = load_boston()
# x = datasets.data
# y = datasets.target 
x, y = datasets.data, datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66)

   
scaler = StandardScaler() 
# scaler.fit(x_train)                    
# x_train = scaler.transform(x_train)     
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)  


#2. 모델구성
input_01 = Input(shape=(13,))
dense_01 = Dense(14)(input_01)
dense_02 = Dense(20, activation='relu')(dense_01)
drop_01 = Dropout(0.2)(dense_02)
dense_03 = Dense(30, activation='relu')(drop_01)
drop_02 = Dropout(0.2)(dense_03)
dense_04 = Dense(20, activation='relu')(drop_02)
dense_05 = Dense(10, activation='relu')(dense_04)
output_01 = Dense(1)(dense_05)
model = Model(inputs=input_01, outputs=output_01)

"""
model = Sequential()
model.add(Dense(64, input_dim=13)) 
model.add(Dropout(0.3)) # 위의 30프로 만큼 dropout                 무작위로 노드를 삭제시키는 것이 dropout
model.add(Dense(32, activation='relu'))                           # 평가 예측에는 dropout이 적용 X
model.add(Dropout(0.2)) # 위의 20프로 만큼 dropout   
model.add(Dense(13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()
"""

start_time = time.time()

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
                      save_best_only=True,                                          # 가장 좋은값을 저장한다
                      filepath="".join([filepath, 'k24_', date, '_', filename])     #      "".join() 괄호 안을 하나의문자로 만든다
                    # filepath='./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5' 
                      )

hist = model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1, callbacks=[earlyStopping, mcp], validation_split=0.2)





# 4. 평가, 예측
print('======================================= 1. 기본 출력 =======================================')
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# loss :  11.482247352600098
# r2스코어 :  0.8626245121807405

# loss :  13.23200798034668
# r2스코어 :  0.8416900670657803
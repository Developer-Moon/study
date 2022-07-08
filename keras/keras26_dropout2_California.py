from sklearn.datasets import fetch_california_housing  
import numpy as np
from sklearn import metrics         
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler    
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Input      
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score 
import time


#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #


#1. 데이터 
datasets = fetch_california_housing()
x = datasets.data  
y = datasets.target 

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )


# scaler = MinMaxScaler() 
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)    
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)       

print(np.min(x_train))  
print(np.max(x_train))  
print(np.min(x_test)) 
print(np.max(x_test))  



#2. 모델구성
input_01 = Input(shape=(8,))
dense_01 = Dense(50)(input_01)
dense_02 = Dense(70, activation='relu')(dense_01)
dropout_01 = Dropout(0.4)(dense_02)
dense_03 = Dense(80, activation='relu')(dropout_01)
dense_04 = Dense(50, activation='relu')(dense_03)
dense_05 = Dense(10, activation='relu')(dense_04)
dense_06 = Dense(10, activation='relu')(dense_05)
dense_07 = Dense(10, activation='relu')(dense_06)
dense_08 = Dense(10, activation='relu')(dense_07)
dropout_02 = Dropout(0.4)(dense_08)
dense_09 = Dense(10, activation='relu')(dropout_02)
dense_10 = Dense(10, activation='relu')(dense_09)
dense_11 = Dense(10, activation='relu')(dense_10)
dense_11 = Dense(10, activation='relu')(dense_10)
dense_12 = Dense(10, activation='relu')(dense_11)
dropout_03 = Dropout(0.2)(dense_12)
dense_13 = Dense(10, activation='relu')(dropout_03)
dense_14 = Dense(10, activation='relu')(dense_13)
dense_15 = Dense(10, activation='relu')(dense_14)
dense_16 = Dense(10, activation='relu')(dense_15)
dense_17 = Dense(10, activation='relu')(dense_16)
output_01 = Dense(1)(dense_17)
model = Model(inputs=input_01, outputs=output_01)



""" 기존 Sequential모델
model = Sequential()
model.add(Dense(50, input_dim=8))
model.add(Dense(70, activation='relu'))2    
model.add(Dense(80, activation='relu'))3
model.add(Dense(50, activation='relu'))4
model.add(Dense(10, activation='relu'))5
model.add(Dense(10, activation='relu'))6
model.add(Dense(10, activation='relu'))7
model.add(Dense(10, activation='relu'))8
model.add(Dense(10, activation='relu'))9
model.add(Dense(10, activation='relu'))10
model.add(Dense(10, activation='relu'))11
model.add(Dense(10, activation='relu'))12
model.add(Dense(10, activation='relu'))13
model.add(Dense(10, activation='relu'))14
model.add(Dense(10, activation='relu'))15
model.add(Dense(10, activation='relu'))16
model.add(Dense(10, activation='relu'))17
model.add(Dense(1))
model.summary() # Total params: 15,591
"""


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                              
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# loss :  [0.2673729360103607, 0.3355602025985718]   기존
# r2스코어 : 0.8082063054489048

# loss :  [0.9301130175590515, 0.6812400817871094] dropout
# r2스코어 : 0.33280505872906796
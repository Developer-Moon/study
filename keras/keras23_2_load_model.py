from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense  
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score 
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66)

scaler = MinMaxScaler()     
# scaler = StandardScaler() 
# scaler = MaxAbsScaler()                                                                                  
# scaler = RobustScaler()
scaler.fit(x_train)                    
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)  


#2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_dim=13))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(13, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))
# model.summary()


# model.save('./_save/keras23_1_save_model.h5') 
model = load_model('./_save/keras23_1_save_model.h5')    # 위의 #2 모델 불러오기 
model.summary()

'''
odel: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                896
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 13)                429
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 112
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 9
=================================================================
Total params: 3,526
Trainable params: 3,526
Non-trainable params: 0
'''






start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')             
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)                                                   
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, callbacks=[earlyStopping], validation_split=0.2)


                                                                

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) # loss :  49.679412841796875    val_loss: 65.4229



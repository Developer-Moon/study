from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense  
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
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()


start_time = time.time()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')             
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint    
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)                                                   
mcp = ModelCheckpoint(moniter='val_loss', mode='auto', verbose=1,
                      save_best_only=True,                                        # 가장 좋은값을 저장한다
                      filepath='./_ModelCheckpoint/keras24_ModelCheckPoint.hdf5'   # 저장경로
                      )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=1, callbacks=[earlyStopping, mcp], validation_split=0.2)



# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

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
                      save_best_only=True,                                          # 가장 좋은값을 저장한다
                      filepath='./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5'   # 저장경로
                      )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=1, callbacks=[earlyStopping, mcp], validation_split=0.2)

model.save('/_save/keras24_3_save_model.h5')


# 4. 평가, 예측
print('======================================= 1. 기본 출력 =======================================')
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# loss :  10.470861434936523
# r2스코어 :  0.8747249028707784


# model = load_model('./_modelCheckPoint/keras24_ModelCheckPoint.hdf5')
# loss :  10.470861434936523
# r2스코어 :  0.8747249028707784
print('======================================= 2. load_model 출력 =======================================')
model2 = load_model('/_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss : ', loss2) 

y_predict2 = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict2)
print('r2스코어 : ', r2)

print('======================================= 3. ModelCheckpoint 출력 =======================================')
model3 = load_model('./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss : ', loss3) 

y_predict3 = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict3)
print('r2스코어 : ', r2)

# ======================================= 1. 기본 출력 =======================================
# 4/4 [==============================] - 0s 665us/step - loss: 9.7148
# loss :  9.714811325073242
# r2스코어 :  0.8837703989724938
# ======================================= 2. load_model 출력 =======================================
# 4/4 [==============================] - 0s 0s/step - loss: 9.7148
# loss :  9.714811325073242
# r2스코어 :  0.8837703989724938
# ======================================= 3. ModelCheckpoint 출력 =======================================
# 4/4 [==============================] - 0s 0s/step - loss: 9.7148
# loss :  9.714811325073242
# r2스코어 :  0.8837703989724938
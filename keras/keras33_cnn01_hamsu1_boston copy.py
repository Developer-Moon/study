from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input 
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score 
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  


#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets.target 
print(x.shape, y.shape) # (506, 13) (506,)


x = x.reshape(506, 13, 1, 1)
print(x.shape)          # (506, 13, 1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)




#2. 모델구성
input_01 = Input(shape=(13, 1, 1))
conv2D_01 = Conv2D(32, (1,1), padding='valid', activation='relu')(input_01)
conv2D_02 = Conv2D(32, (1,1), activation='relu')(conv2D_01)
conv2D_03 = Conv2D(32, (1,1), activation='relu')(conv2D_02)
conv2D_04 = Conv2D(32, (1,1), activation='relu')(conv2D_03)
flatten_01 = Flatten()(conv2D_04)
dense_01 = Dense(32, activation='relu')(flatten_01)
dropout_01 = Dropout(0.2)(dense_01)
dense_02 = Dense(32, activation='relu')(dropout_01)
output_01 = Dense(1)(dense_02)
model = Model(inputs=input_01, outputs=output_01)






# model = Sequential()                                                                             
# model.add(Conv2D(filters=32, kernel_size=(1,1), input_shape=(13, 1, 1))) # 최대넓이가 가로13 세로 1이라 커널 사이즈 최대가 (1, 1)이 된다
# model.add(Conv2D(32, (1, 1), padding='valid', activation='relu'))    
# model.add(Conv2D(32, (1, 1), activation='relu'))    
# model.add(Conv2D(32, (1, 1), activation='relu'))    
# # model.add(MaxPooling2D())    #(14, 14, 64)                             # MaxPooling2D 도 최대넓이가 가로13 세로 1이라 사용 불가능 사용시 에러
# model.add(Flatten()) # (N, 252)                                                                         
# model.add(Dense(32, activation='relu'))    
# model.add(Dropout(0.2))                                                                                                     
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1))
# model.summary()



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')             
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint  
earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, verbose=1, callbacks=[earlyStopping], validation_split=0.2)




# 4. 평가, 예측
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


# CNN모델
# loss :  25.453134536743164
# r2스코어 :  0.6954745082749104
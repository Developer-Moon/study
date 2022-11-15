from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.datasets import load_wine
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

import tensorflow as tf            
tf.random.set_seed(66)


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target 
print(x.shape) # (178, 13)
print(y.shape) # (178,)
# print(np.unique(x))
print(np.unique(y, return_counts=True))   

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
print(y)
print(y.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )


scaler = MinMaxScaler() 
scaler.fit(x_train)     # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)        # train이 변한 범위에 맞춰서 변환됨

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위



#2. 모델구성
input_01 = Input(shape=(13,))
dense_01 = Dense(100)(input_01)
dense_02 = Dense(100)(dense_01)
dense_03 = Dense(100)(dense_02)
dense_04 = Dense(100)(dense_03)
dense_05 = Dense(100)(dense_04)
dense_06 = Dense(100)(dense_05)
dense_07 = Dense(100)(dense_06)
dense_08 = Dense(100)(dense_07)
dense_09 = Dense(100)(dense_08)
dense_10 = Dense(100)(dense_09)
dense_11 = Dense(100)(dense_10)
output_01 = Dense(3, activation='softmax')(dense_11)
model = Model(inputs=input_01, outputs=output_01)

# model.save('./_save/keras23_7_06_wine_01_save_model.h5')          
# model.save_weights('./_save/keras23_7_06_wine_03_save_weights.h5')   
         
# model = load_model('./_save/keras23_7_06_wine_01_save_model.h5')        
# model.load_weights('./_save/keras23_7_06_wine_03_save_weights.h5')        
model.load_weights('./_save/keras23_7_06_wine_04_save_weights.h5')   
                           
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                                            
# from tensorflow.python.keras.callbacks import EarlyStopping      
# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
# hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, callbacks=[earlyStopping], verbose=1) 


# model.save('./_save/keras23_7_06_wine_02_save_model.h5') 
# model.save_weights('./_save/keras23_7_06_wine_04_save_weights.h5')

# model = load_model('./_save/keras23_7_06_wine_02_save_model.h5')  


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 

from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)


# loss :  0.09179487079381943 --------------------- 기존훈련     
# acc스코어 : 0.9491525423728814

# loss :  0.09179487079381943 --------------------- 01_save_model
# acc스코어 : 0.9491525423728814

# loss :  0.09179487079381943 --------------------- 02_save_model : 모델과 가중치 같이 저장          [기존 훈련의 가장 좋은 값과 같다]
# acc스코어 : 0.9491525423728814

# loss :  1.1005196571350098 ---------------------- 03_save_weights : 랜덤 가중치 저장
# acc스코어 : 0.3728813559322034

# loss :  0.09179487079381943 --------------------- 04_save_weights : 훈련된 가장 좋게 저장된 가중치  [기존 훈련의 가장 좋은 값과 같다]
# acc스코어 : 0.9491525423728814






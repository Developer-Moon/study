from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM
from sklearn.model_selection import train_test_split        
from sklearn.datasets import fetch_covtype                            
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd 
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('쥐피유 돌아유')
    aaa = 'gpu - 쥐피유 돌아유'
else:
    print('내가 돌아유')
    aaa = 'cpu - 내가 돌아유'




#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

import tensorflow as tf            
tf.random.set_seed(66)         #텐서플로의 난수표 66번 사용 이 난수는 w = 웨이트

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))   

x = x.reshape(581012, 54, 1)

y = pd.get_dummies(y)
print(pd.get_dummies(y))
print(y.shape)

print(np.unique(y, return_counts=True)) 

"""
from tensorflow.keras.utils import to_categorical # 범주
y = to_categorical(y)
print(y)
print(y.shape) # (581012, 8)                  !!!!!!!!!!!!!!!!! - 갑자기 여기서는 왜 8개가 나오나??
"""

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )




#2. 모델구성
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(54, 1))) # 최대넓이가 가로13 세로 1이라 커널 사이즈 최대가 (1, 1)이 된다                                                                          
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))




#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',            # 다중분류의 loss는 $$$당!분!간$$$ categorical_crossentropy 만 쓴다 (20220704)
              optimizer='adam',
              metrics=['accuracy'])
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=20, batch_size=1000, validation_split=0.2, callbacks=[earlyStopping], verbose=1) 


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 


from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)

y_test = tf.argmax(y_test, axis=1)
print(y_test)




# r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)  
print(aaa, 'acc스코어 :', acc)


# loss :  0.6418493986129761        기존훈련
# accuracy :  0.7204773426055908

# loss :  1.9459669589996338          dropout
# accuracy :  0.36302897334098816


# CNN 모델
# loss :  0.7970449924468994
# accuracy :  0.6454358696937561

# LSTM
# loss :  1.2055710554122925
# accuracy :  0.4888908565044403
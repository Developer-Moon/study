from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM
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
print(np.unique(y, return_counts=True))     # [0 1 2] - (array([0, 1, 2]), array([59, 71, 48], dtype=int64)) 0이 59개  1이 71개  2가 48개

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
print(y)
print(y.shape) # (150, 3)

x = x.reshape(178, 13, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )






#2. 모델구성
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(13, 1))) # 최대넓이가 가로13 세로 1이라 커널 사이즈 최대가 (1, 1)이 된다                                                                          
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))




                           
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',            # 다중분류의 loss는 $$$당!분!간$$$ categorical_crossentropy 만 쓴다 (20220704)
              optimizer='adam',
              metrics=['accuracy'])
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1) 


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


# loss :  0.09179487079381943       기존훈련
# accuracy :  0.9491525292396545

# loss :  1.1102482080459595        dropout
# accuracy :  0.32203391194343567

# loss :  14.469804763793945
# accuracy :  0.5932203531265259

# LSTM
# loss :  0.689313530921936
# accuracy :  0.7796609997749329
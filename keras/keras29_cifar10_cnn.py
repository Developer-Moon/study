from tensorflow.python.keras.models import Sequential, Model, load_model                             # 컬러 - softmax 100개
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input  
from sklearn.model_selection import train_test_split        
from keras.datasets import mnist,  cifar10
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd 


import tensorflow as tf            
tf.random.set_seed(66)     

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1) 
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True)) # y값의 라벨확인
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]))


from tensorflow.keras.utils import to_categorical # 범주
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train)
print(y_test)
print(y_train.shape) # (581012, 8)                
print(y_test.shape)



#2. 모델구성CNN모델                                                                                         
model = Sequential()                                                                             
model.add(Conv2D(filters=64, kernel_size=(2,2),
                 padding='same',            # padding 0이라는 패딩을 씌워서 이미지를 조각낼때 가장자리 부분을 두번 이상 넣어줘서 다른 부분보다 덜 학습되는걸 방지 
                 input_shape=(32, 32, 3)))  # 통상 shape를 다음 레이어에도 유지하고 싶을때 padding을 쓴다                                                                                    
model.add(Conv2D(32, (2, 2), padding='valid', activation='relu'))    # 디폴트
model.add(Conv2D(32, (2, 2), activation='relu'))    # 디폴트
model.add(Conv2D(32, (2, 2), activation='relu'))    # 디폴트
model.add(MaxPooling2D())    #(14, 14, 64) 
model.add(Flatten()) # (N, 252)                                                                         
model.add(Dense(32, activation='relu'))    
model.add(Dropout(0.2))                                                                                                     
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary() 


# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # True 또는 False, 양성 또는 음성 등 2개의 클래스를 분류할 수 있는 분류기를 의미  
                                                                                                                                                                                                                                                      
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint      
import datetime

date = datetime.datetime.now()    
print(date)                        # 2022-07-07 17:25:03.261773
date = date.strftime('%m%d_%H%M') 
print(date)                        # 0707_1724

filepath = './_ModelCheckpoint/k25/'
filename = '{epoch:04d}-{loss:.4f}.hdf5'   
#                   4번재 자리      소수 4번째 자리
################################

earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)     
mcp = ModelCheckpoint(moniter='val_loss', mode='auto', verbose=1,
                      save_best_only=True,                                          # 가장 좋은값을 저장한다
                      filepath="".join([filepath, 'car10', date, '_', filename])     #      "".join() 괄호 안을 하나의문자로 만든다
                    # filepath='./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5' 
                      )   

  
hist = model.fit(x_train, y_train, epochs=200, batch_size=100, validation_split=0.2, callbacks=[earlyStopping, mcp], verbose=1)  



#4. 결과, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 


from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)

y_test = tf.argmax(y_test, axis=1)
print(y_test)


acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)

# loss :  1.1806913614273071
# accuracy :  0.5965999960899353
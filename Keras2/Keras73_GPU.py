import tensorflow as tf
# 선택적 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus : 
    try : # 예외처리
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU') # 보이는 장비가 GPU의 0번째
        # tf.config.experimental.set_visible_devices(gpus[1], 'GPU') # 보이는 장비가 GPU의 1번째
        # tf.config.experimental.set_visible_devices([gpus[0], gpus[1]], 'GPU') # 2개 다 넣는다고 다 도는게 아니다
        # tf.config.experimental.set_visible_devices(gpus[0], 'CPU')
        # GPU가 2개 이상일때 하나의 파일을 2개 이상 같이 훈련할 수 있다 + CPU까지
        
    except RuntimeError as e :
        print(e)



from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D 
from tensorflow.python.keras.models import Sequential
from keras.datasets import mnist,cifar10,cifar100,fashion_mnist
import pandas as pd
import numpy as np

import tensorflow as tf            
tf.random.set_seed(66)

#성능은 CNN 보다 좋게

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) (50000, 1) 
print(x_test.shape, y_test.shape)    # (10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True)) # y값의 라벨확인
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]))


x_train = x_train.reshape(50000, 32*32*3, 1)
x_test = x_test.reshape(10000, 32*32*3, 1)
print(x_train.shape) # (50000, 3072)
print(x_test.shape)  # (10000, 3072)


from tensorflow.python.keras.utils.np_utils import to_categorical # 범주
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000,3072,1)
x_test = x_test.reshape(10000,3072,1)

#2.모델구성
model = Sequential()
model.add(Conv1D(50,2,activation='relu', input_shape=(3072,1))) 
model.add(Flatten())
model.add(Dense(50,activation= 'relu'))
model.add(Dense(50,activation= 'relu'))
model.add(Dense(50,activation= 'relu'))
model.add(Dense(10,activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint      
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)        

  
hist = model.fit(x_train, y_train, epochs=100, batch_size=500, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  
                                            # batch_size:32 디폴트값 3번정도 말 한듯




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


# CNN 사용
# loss :  1.0057822465896606
# accuracy :  0.6510000228881836

# LSTM
# loss :  2.3080384731292725
# accuracy :  0.1137000024318695

# Conv1D 
# loss :  2.3025946617126465
# accuracy :  0.10000000149011612
import tensorflow as tf
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


# strategy = tf.distribute.MirroredStrategy() # 분산처리
'''
strategy = tf.distribute.MirroredStrategy( 
    # cross_device_ops = tf.distribute.HierarchicalCopyAllReduce() # 성능차이 비교하기
    cross_device_ops = tf.distribute.ReductionToOneDevice() 
) 
'''
'''
strategy = tf.distribute.MirroredStrategy(
    # device=['/gpu:0']
    # devicecs=['/gpu:1']
    # devicecs=['/cpu']
    # devicecs=['/cpu', '/gpu:0']
    # devicecs=['/cpu', '/gpu:1']
    # devicecs=['/cpu', '/gpu:0', '/gpu:1']
    # devicecs=['/gpu:0', '/gpu:1'] # 에러
)
'''

# strategy = tf.distribute.experimental.CentralStorageStrategy()
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    # tf.distribute.experimental.CollectiveCommunication.RING
    # tf.distribute.experimental.CollectiveCommunication.NCCL
    tf.distribute.experimental.CollectiveCommunication.AUTO
    
)
 


with strategy.scope() :

#2.모델구성
    model = Sequential()
    model.add(Conv1D(50,2,activation='relu', input_shape=(3072,1))) 
    model.add(Flatten())
    model.add(Dense(50,activation= 'relu'))
    model.add(Dense(10,activation='softmax'))

    #3. 컴파일, 훈련
    model.compile(loss='categorical_crossentropy', optimizer='adam') 
    # from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint      
    # earlyStopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)        
    
model.fit(x_train, y_train, epochs=10, batch_size=500, validation_split=0.2, verbose=1)  
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
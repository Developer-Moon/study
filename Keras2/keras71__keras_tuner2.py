from keras.datasets import mnist
import tensorflow as tf
# print(tf.__version__) # 2.10.0
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizer_v2.adam import Adam
from sklearn.metrics import accuracy_score
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255., x_test/255.


def get_model(hp) :
    hp_unit1 = hp.Int('units1', min_value=16, max_value=512, step=16) # 인트형으로 16부터 512 사이의 16스텝 
    hp_unit2 = hp.Int('units2', min_value=16, max_value=512, step=16) # 16, 512 노드의 개수
    hp_unit3 = hp.Int('units3', min_value=16, max_value=512, step=16) # 이걸 Dense안에 넣는다
    hp_unit4 = hp.Int('units4', min_value=16, max_value=512, step=16)

    hp_drop1 = hp.Choice('dropout1', values=[0.0, 0.2, 0.3, 0.4, 0.5]) # dropout을 여기서 고를꺼다
    hp_drop2 = hp.Choice('dropout2', values=[0.0, 0.2, 0.3, 0.4, 0.5]) 
    
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(hp_unit1, activation='relu')) # 활성화함수도 파라미터로 잡아보자
    model.add(Dropout(hp_drop1))
    
    model.add(Dense(hp_unit2, activation='relu')) 
    model.add(Dropout(hp_drop1))
    
    model.add(Dense(hp_unit3, activation='relu'))
    model.add(Dropout(hp_drop2))
    
    model.add(Dense(hp_unit4, activation='relu'))
    model.add(Dropout(hp_drop2))
    
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=hp_lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
    
kerastuner = kt.Hyperband(get_model, # 다시 훈련하면 훈련이 안된다
                          directory='my_dri',       
                          objective='val_accuracy', # objective 기준치
                          max_epochs=6,             # 최대 6번 내에서 돌리자 (2번에서 6번 사이 랜덤으로 돈다)
                          project_name='kerastuner-mnist') # 이름, (가중치와 파라미터가 저장된다, 이름바꿔서 저장) 

kerastuner.search(x_train, y_train, validation_data=(x_test, y_test), batch_size=32) # 데이터 연결 
                                                                                     # batch 까지 서치하려면 class로 정의 찾아보자

best_hps = kerastuner.get_best_hyperparameters(num_trials=2)[0] # num_trials 몇번할껀지 어째할껀지-알아서 공부

print('best parameter - units1 :', best_hps.get('units1'))
print('best parameter - units2 :', best_hps.get('units2'))
print('best parameter - units3 :', best_hps.get('units3'))
print('best parameter - units4 :', best_hps.get('units4'))

print('best parameter - dropout1 :', best_hps.get('dropout1'))
print('best parameter - dropout2 :', best_hps.get('dropout2'))

print('best parameter - learning_rate :', best_hps.get('learning_rate'))

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, factor=0.5) # factor 러닝레이트 50프로 감소

model = kerastuner.hypermodel.build(best_hps) 
history = model.fit(x_train, y_train, callbacks=[es, reduce_lr], validation_split=0.2, epochs=300)

loss, accuracy = model.evaluate(x_test, y_test)
print('accuracy :', accuracy)
print('loss :', loss)

# y_predict = model.predict(x_test)
# y_predict = np.argmax(y_predict, axis=1) 
# y_test = np.argmax(y_test, axis=1) 

# acc_score = accuracy_score(y_test, y_predict)
# print('acc_score :', acc_score)
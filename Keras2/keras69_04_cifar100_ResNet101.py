from keras.applications.resnet import preprocess_input, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.applications import ResNet101
from keras.datasets import cifar100
import numpy as np
from keras.layers import GlobalAveragePooling2D


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델구성
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
x = base_model.output # 모델의 가장 끝 부분을 x
x = GlobalAveragePooling2D()(x)
x = Dense(102, activation='relu')(x)
output_01 = Dense(100, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_01)


#3. 컴파일
from keras.optimizers import Adam
learning_rate = 0.01
optimizer = Adam(lr=learning_rate)

model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau # 

es= EarlyStopping(monitor='val_loss', patience=20, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5) # factor=0.5 : 50% 만큼 lr을 감소 시킨다  디폴트 lr은 0.001
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4, batch_size=128, callbacks=[es, reduce_lr])
end = time.time() - start


#4.결과, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_test)
print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc스코어: ', acc) # acc스코어:  0.1477
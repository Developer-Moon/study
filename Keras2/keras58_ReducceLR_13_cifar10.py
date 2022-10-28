from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# 1. data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape) # (50000, 32, 32, 3) (10000, 32, 32, 3)

x_train = x_train.reshape(50000, 32*32*3) # ValueError: Found array with dim 4. MinMaxScaler expected <= 2. 차원 문제 에러
x_test = x_test.reshape(10000, 32*32*3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3) 
x_test = x_test.reshape(10000, 32, 32, 3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. model
activation='relu'
drop=0.3

inputs = Input(shape=(32, 32, 3), name='input')
x = Conv2D(64, (3, 3), padding='valid', activation=activation, name='hidden1')(inputs) # 27, 27, 128
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding='same', activation=activation, name='hidden2')(x)        # 27, 27, 64
# x = Dropout(drop)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), padding='valid', activation=activation, name='hidden3')(x)       # 25, 25, 32
x = Dropout(drop)(x) #
# x = Flatten()(x)     # 25 * 25 * 32 = 20000
x = GlobalAveragePooling2D()(x)

x = Dense(64, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x) 
outputs = Dense(100, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()



#3. compile
from tensorflow.keras.optimizers import Adam
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


#4.결과예측
loss, acc = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
# y_pred = model.predict(x_test)
# print(y_pred[:10])

# y_pred = np.argmax(model.predict(x_test), axis=-1)

print('걸린 시간: ', end)
print('loss :', loss)
print('acc :', acc)
# print('acc score: ', accuracy_score(y_test, y_pred))

# 걸린 시간:  409.5901219844818
# loss : 3.448904275894165
# acc : 0.16949999332427979
from tkinter import Image
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np  
# 모델구성 및 성능비교 
# 성능비교, 증폭 전 후 비교

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./225,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.01,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

print(x_train.shape) # (60000, 28, 28)


#원본
x_train = x_train.reshape(60000, 28, 28, 1)

print(x_test.shape) # (10000, 28, 28)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # 4차원 변경
              
print(x_train.shape)       # (60000, 28, 28, 1)
print(y_train.shape)       # (60000,)



print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,)




#2.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.metrics import r2_score,accuracy_score

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape = (28, 28, 1),activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))


#3.컴파일,훈련
from tensorflow.python.keras.callbacks import EarlyStopping

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
earlyStopping= EarlyStopping(monitor= 'val_loss', patience=30, mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=32, callbacks=[earlyStopping])



#4.평가,예측 
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)


from sklearn.metrics import r2_score,accuracy_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
print('loss : ', loss)

# r2 : 0.8589658431605158
# loss : 1.16353178024292




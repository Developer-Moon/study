from tkinter import Image
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np  



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

train_datagen2 = ImageDataGenerator(
    
)



augument_size = 40000 # 증가시키다
batch_size = 64
randidx = np.random.randint(x_train.shape[0], size=augument_size)  # 60000개 중에 40000개를 랜덤하게 정수를 뽑는다
                                                                       
x_augumented = x_train[randidx].copy
y_augumented = y_train[randidx].copy

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_augumented = x_augumented.reshape(x_augumented.shape[0], x_augumented.shape[1],
                                    x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

xy_train = train_datagen2.flow(x_train, y_train,
                               batch_size=64, 
                               shuffle=False)

# print(xy_train[0].shape) # 튜플이당
print(xy_train[0][0].shape)
print(xy_train[0][1].shape)

    



#2.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.metrics import r2_score,accuracy_score
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (28,28,1),activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))




#3.컴파일,훈련       
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(xy_train, epochs=10, steps_per_epoch=len(xy_train))

print(len(xy_train))
print(len(xy_train[0]))
print(len(xy_train[0][0]))
print(len(xy_train[0][1]))




#4.평가,예측 
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)




from sklearn.metrics import r2_score,accuracy_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
print('loss : ', loss)

# r2스코어 : 0.8495036103238185
# loss :  1.2415952682495117




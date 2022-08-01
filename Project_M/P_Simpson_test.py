# https://www.kaggle.com/code/everydaycodings/simpsons-image-classification-cnn-val-acc-93
from tensorflow.python.keras.models import Model, load_model, Sequential
from tensorflow.python.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
tf.random.set_seed(9)


# 1. 데이터
scale_datagen = ImageDataGenerator(rescale=1./255)

test_img = scale_datagen.flow_from_directory(
    'd:/study_data/_data/Project_M/20220725_Simpson/test_image/',
    target_size=(150, 150),
    batch_size=10100,
    class_mode='categorical',
    shuffle=True
) 

character = {0 :'abraham_grampa_simpson',
             1 : 'agnes_skinner',
             2 : 'bart_simpson',
             3 : 'homer_simpson',
             4 : 'krusty_the_clown',
             5 : 'lisa_simpson',
             6 : 'maggie_simpson',
             7 : 'marge_simpson',
             8 : 'milhouse_van_houten',
             9 : 'sideshow_bob'}


x_train = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/train_x.npy')
y_train = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/train_y.npy')

x_test = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/test_x.npy')
y_test = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/test_y.npy')

np.save('d:/study_data/_data/Project_M/20220725_Simpson/_npy/test_img.npy', arr =test_img[0][0])
img_test = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/test_img.npy')

print(img_test.shape)
print(img_test[0][0].shape)
print(img_test[0][1].shape)
print(x_train.shape, y_train.shape) # (8048, 150, 150, 3) (8048, 10)
print(x_test.shape, y_test.shape)   # (2012, 150, 150, 3) (2012, 10)



# 2. 모델구성
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(150, 150, 3), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation="relu")) 
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()

# model.load_weights('d:/study_data/_data/Project_M/20220725_Simpson/_save/save_weights_model.h5')

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=70, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=200, batch_size=32, callbacks=[Es], validation_split=0.2)

# model.save_weights('d:/study_data/_data/Project_M/20220725_Simpson/_save/save_weights_model.h5') # 저장된 가중치


#4. 평가, 예측
result = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
img_test_predict = model.predict(img_test)

y_predict = tf.argmax(y_predict, axis=1)
y_test = tf.argmax(y_test, axis=1)
img_test_predict = tf.argmax(img_test_predict, axis=1)


print(img_test_predict)
print('loss : ', result[0])
print('acc  : ', result[1])

# loss :  0.6125920414924622
# acc  :  0.8235586285591125









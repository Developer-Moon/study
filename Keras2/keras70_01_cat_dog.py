from keras.applications.resnet import preprocess_input, decode_predictions
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input
from keras.applications import DenseNet121
from keras.datasets import cifar100
import numpy as np
from keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
x_train = np.load('d:/study_data/_save/_npy/kears_47_01_01_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/kears_47_01_02_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/kears_47_01_03_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/kears_47_01_04_test_y.npy') 

print(x_train.shape) # (8005, 150, 150, 3)
print(x_test.shape)  # (8005,)
print(y_train.shape) # (2023, 150, 150, 3)
print(y_test.shape)  # (2023,)



#2. 모델구성
inputs = DenseNet121(weights='imagenet', include_top=False, shape=(150, 150, 3))
x = Dense(110, activation='relu')(inputs)
x = GlobalAveragePooling2D()(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=outputs)



#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping= EarlyStopping(monitor='val_loss', patience=30, mode='auto', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=30, batch_size=32, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  



#4. 평가, 예측
acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_accuracy : ', val_accuracy[-1])


loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict)

print("loss :", loss)
print("accuracy :", acc)

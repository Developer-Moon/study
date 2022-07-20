from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np



x_train = np.load('d:/study_data/_save/_npy/kears_47_01_train_x.npy')
x_test = np.load('d:/study_data/_save/_npy/kears_47_02_train_y.npy')
y_train = np.load('d:/study_data/_save/_npy/kears_47_03_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/kears_47_04_test_y.npy') 

print(x_train.shape) # (80, 150, 150, 3)
print(x_test.shape)  # (80,)
print(y_train.shape) # (5, 150, 150, 3)
print(y_test.shape)  # (5,)


#2. 모델구성
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))




#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')
earlyStopping= EarlyStopping(monitor= 'val_loss',patience=20,mode='auto',restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  




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
acc = accuracy_score(y_test, y_predict)

print("loss :", loss)
print("accuracy :", acc)


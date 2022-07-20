import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
 
x = np.load('D:\study_data\_save\_npy/kears_47_04_01_x.npy.')
y = np.load('D:\study_data\_save\_npy/kears_47_04_02_y.npy')
moon = np.load('D:\study_data\_save\_npy/kears_47_04_moon.npy')


# datagen = ImageDataGenerator(rescale=1./255)

# moon = datagen.flow_from_directory(
#     'D:\study_data\_data\image\_moon\moon', # 이 경로의 이미지파일을 불러 수치화
#     target_size=(150,150),# 크기들을 일정하게 맞춰준다.
#     batch_size=30,
#     class_mode='binary', 
#     # color_mode='grayscale', #디폴트값은 컬러
#     shuffle=True,
#     )

print(moon[0][0].shape)
print(moon[0][1].shape)
moon = moon[0][0]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.7,shuffle=True,random_state=70)


#2.모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten
from sklearn.metrics import r2_score,accuracy_score
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape = (150,150,3),activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3.컴파일,훈련
model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# batch를 최대로 잡으면 이렇게도 가능.
from tensorflow.python.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor= 'val_loss',patience=30,mode='auto',restore_best_weights=True)
hist = model.fit(x_train,y_train,epochs=10,validation_split=0.05,verbose=1,batch_size=32,callbacks=es)
# hist = model.fit_generator(x_train,y_train, epochs= 40, steps_per_epoch=32,
#                                         #전체데이터/batch = 160/5 = 32
#                     validation_data=x_train,
#                     validation_steps=4) #val_steps: 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 

accuracy = hist.history['accuracy']
val_accuracy =  hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_accuracy : ', val_accuracy[-1])


#4.평가,예측 
loss = model.evaluate(x_test,y_test)


y_predict = model.predict(moon)
print(y_predict)
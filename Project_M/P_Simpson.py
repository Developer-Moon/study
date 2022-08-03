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

character = {0 :'abraham_grampa_simpson', # 남 - 900
             1 : 'agnes_skinner',         # 여 - 40
             2 : 'bart_simpson',          # 남 - 1300
             3 : 'edna_krabappel',        # 여 - 457
             4 : 'homer_simpson',         # 남 - 2200
             5 : 'krusty_the_clown',      # 남 - 1200
             6 : 'lisa_simpson',          # 여 - 1300
             7 : 'maggie_simpson',        # 여 - 120
             8 : 'marge_simpson',         # 여 - 1200
             9 : 'milhouse_van_houten'}   # 남 - 1000

x_train = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/train_x.npy')
y_train = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/train_y.npy')

x_test = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/test_x.npy')
y_test = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/test_y.npy')

np.save('d:/study_data/_data/Project_M/20220725_Simpson/_npy/test_img.npy', arr = test_img[0][0])
img_test = np.load('d:/study_data/_data/Project_M/20220725_Simpson/_npy/test_img.npy')

print(img_test.shape)
print(img_test[0][0].shape)
print(img_test[0][1].shape)
print(x_train.shape, y_train.shape) # (8048, 150, 150, 3) (8048, 10)
print(x_test.shape, y_test.shape)   # (2012, 150, 150, 3) (2012, 10)



# 2. 모델구성
"""
input_01 = Input(shape=(150, 150, 3))
conv_01 = Conv2D(64,(3,3), activation='relu')(input_01) # , padding='same'
maxpool_01 = MaxPool2D()(conv_01)
conv_02 = Conv2D(64,(3,3), activation='relu')(maxpool_01)
maxpool_02 = MaxPool2D()(conv_02)
conv_03 = Conv2D(64,(3,3), activation='relu')(maxpool_02)
maxpool_03 = MaxPool2D()(conv_03)
flattin = Flatten()(maxpool_03)
output = Dense(10, activation='softmax')(flattin)
model = Model(inputs=input_01, outputs=output)
model.summary()
"""
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
Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=True)
log = model.fit(x_train, y_train, epochs=300, batch_size=32, callbacks=[Es], validation_split=0.2)

model.save_weights('d:/study_data/_data/Project_M/20220725_Simpson/_save/save_weights_model.h5') # 저장된 가중치


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

# loss :  0.557797372341156
# acc  :  0.8339959979057312

f = '저는'
b = '입니당!'

if img_test_predict == 0 :
    print(f, 'abraham_grampa_simpson', b)
elif img_test_predict == 1 :
     print(f, 'agnes_skinner', b)
elif img_test_predict == 2 :
     print(f, 'bart_simpson', b)
elif img_test_predict == 3 :
     print(f, 'edna_krabappel', b)
elif img_test_predict == 4 :
     print(f, 'homer_simpson', b)
elif img_test_predict == 5 :
     print(f, 'krusty_the_clown', b)
elif img_test_predict == 6 :
     print(f, 'lisa_simpson', b)
elif img_test_predict == 7 :
     print(f, 'maggie_simpson', b)
elif img_test_predict == 8 :
     print(f, 'marge_simpson', b)
elif img_test_predict == 9 :
     print(f, 'milhouse_van_houten', b)     
                           
         
          
          
          
          
          
                              
import matplotlib.pyplot as plt    
plt.figure(figsize=(9,6))                                                   # 판 크기
plt.plot(log.history['accuracy'], marker='.', c='red', label='accuracy')           # marker=로스부분 .으로 표시   c='red' 그래프를 붉은컬러로  label='loss' 이그래프의 이름(label)은 loss
plt.plot(log.history['val_accuracy'], marker='.', c='blue', label='val_accuracy') 


plt.grid()           
plt.title('accuracy') 
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.legend(loc='upper right') #   label값이 레전드에 명시가 되며 이걸 우측상단에 올린다 location = loc            위치값 upper right', 'lower left', 'center left', 'center 이런게 있다
plt.legend() # 자동으로 빈 공가넹 표시
plt.show()                              

     # marge_simpson 










import numpy as np          
from keras.preprocessing.image import ImageDataGenerator


x_train = np.load('d:/study_data/_save/_npy/kears_46_05_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/kears_46_05_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/kears_46_05_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/kears_46_05_test_y.npy')


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)




# #1. 데이터
# # train_datagen = ImageDataGenerator(  # ImageDataGenerator - 이미지를 숫자화 이 옵션들은 증폭작업에 사용 (임의로 이중 1개만 랜덤으로 적용된다)
# #    rescale=1./225, 
#     # horizontal_flip=True,   # 수평방향 뒤집기  뒤집으면 2개가 된다   
#     # vertical_flip=True,     # 수직 방향 뒤집기 
#     # width_shift_range=0.1,  # 좌우이동
#     # height_shift_range=0.1, # 상하이동
#     # rotation_range=5,       # 돌리기
#     # zoom_range=1.2,         # 확대
#     # shear_range=0.7,        # 이미지 기울기(확인 필요, 3차원 기울기??)
#     # fill_mode='nearest'     # 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
# )



# # test는 이렇게 할거라는걸 셋팅만 한 상태
# test_datagen = ImageDataGenerator( # 평가 데이터는 증폭시키지 않는다, 확인해야하는 값은 그대로 사용하기 떄문 
#     rescale=1./255
# )


# xy_train = train_datagen.flow_from_directory(                # flow_from_directory - 폴더(directory)에서 가져온다
#         'D:/study_data/_data/image/brain/train/',
#         target_size=(150, 150), # 수집한 image의 크기가 다 다르니까 크기를 일정하게 해준다 - 가로세로가 일정한 비율이 아니라면?
#         batch_size=500,
#         class_mode='binary',                                  # binary(이진법) - 분류하는게 0 or 1이라서  다중분류라면?
#         shuffle=True,
#         color_mode='grayscale'  # 디폴트값은 컬러
#         # Found 160 images belonging to 2 classes.
#     )

# xy_test = test_datagen.flow_from_directory(                # flow_from_directory - 폴더(directory)에서 가져온다
#         'D:/study_data/_data/image/brain/test/',
#         target_size=(150, 150), # 수집한 image의 크기가 다 다르니까 크기를 일정하게 해준다 - 가로세로가 일정한 비율이 아니라면?
#         batch_size=500,
#         class_mode='binary',                                  # binary(이진법) - 분류하는게 0 or 1이라서  다중분류라면?
#         shuffle=True,
#         color_mode='grayscale'
#         # Found 160 images belonging to 2 classes.
#     )

# print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x0000017DD2D74F70>

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

# print(xy_train[31]) # array([1., 1., 0., 1., 0.], dtype=float32))   y갑이 5개 batch_size가 5??  x, y가 같이 포함 되어있다

# 160개의 데이터가 배치 5개 단위로 짤렸고 5개 단위로 잘린게 32개다 
# print(xy_train[33]) 를 쓰면 에라

# print(xy_train[0])   # xy와 다 같이 들어있다
# print(xy_train[0][0])       # 마지막 배치
# print(xy_train[0][0].shape) # x만 나온다
# print(xy_train[0][1]) # 
# print(xy_train[31][2]) # 에러 0과 1만 나오니까


# print(xy_train[0][0].shape, xy_train[0][1].shape)
# print(xy_test[0][0].shape, xy_test[0][1].shape)

# np.save('d:/study_data/_save/kears_46_05_train_x.npy', arr=xy_train[0][0]) # train_x이 numpy파일로 저장
# np.save('d:/study_data/_save/kears_46_05_train_y.npy', arr=xy_train[0][1]) # train_y
# np.save('d:/study_data/_save/kears_46_05_test_x.npy', arr=xy_test[0][0])  # test_x
# np.save('d:/study_data/_save/kears_46_05_test_x.npy', arr=xy_test[0][1])  # test_y




# 현재 5, 200, 200, 1 짜리 데이턱 32개다















#2. 모델구성
"""
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 1), activation='relu'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(xy_train[0][0]) # x와 y가 하나로 되어있다 배치사이즈를 최대로 잡으면 이거도 가능
hist = model.fit_generator(xy_train, epochs=10, steps_per_epoch=32,      # 통상적으로 전체데이터/batch = 160/5 = 32   fit_generator에는 batch가 없다 위의 generator에 정의 해놓음
                    validation_data=xy_test, validation_steps=4)  # fit_generator 이걸 쓰면 안에 한번에 훈련가능


#4. 평가, 예측
acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_accuracy : ', val_accuracy[-1])

# loss :  0.7061700224876404
# val_loss :  0.5988134145736694
# acc :  0.5249999761581421
# val_accuracy :  0.800000011920929

# 그림그리기

import matplotlib.pyplot as plt    
plt.figure(figsize=(9,9))                                                   
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')           
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 
plt.plot(hist.history['accuracy'], marker='.', c='green', label='accuracy') 
plt.plot(hist.history['val_accuracy'], marker='.', c='yellow', label='val_accuracy') 



plt.grid()      
plt.title('안결바보')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.legend(loc='upper right') #   label값이 레전드에 명시가 되며 이걸 우측상단에 올린다 location = loc            위치값 upper right', 'lower left', 'center left', 'center 이런게 있다
plt.legend()
plt.show()

"""
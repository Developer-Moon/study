from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
tf.random.set_seed(9)


# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=5,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
    )

scale_datagen = ImageDataGenerator(rescale=1./255)

xy = scale_datagen.flow_from_directory(
    'd:/study_data/_data/Project_M/test/_images/',
    target_size=(150, 150),
    batch_size=10100,
    class_mode='categorical',
    shuffle=False
) # print(xy) - Found 10478 images belonging to 10 classes.

print(xy.class_indices)

# 파일 불러온 변수에서 xy 분리
x_train = xy[0][0]
y_train = xy[0][1]
print(x_train.shape, y_train.shape) # (1760, 150, 150, 3) (1760, 10)

randidx_01 = x_train[1600:1640]
augument_size_01 = 160
randidx_01 = np.random.randint(randidx_01.shape[0], size=augument_size_01)
x_augument_01 = x_train[randidx_01].copy()
y_augument_01 = y_train[randidx_01].copy()
x_augument_01 = train_datagen.flow(x_augument_01, y_augument_01, batch_size=augument_size_01, shuffle=False).next()[0]

randidx_02 = x_train[1640:1760]
augument_size_02 = 80 
randidx_02 = np.random.randint(randidx_02.shape[0], size=augument_size_02)
x_augument_02 = x_train[randidx_02].copy()
y_augument_02 = y_train[randidx_02].copy()
x_augument_02 = train_datagen.flow(x_augument_02, y_augument_02, batch_size=augument_size_01, shuffle=False).next()[0]

# 원본train과 증폭train 합치기
x_train = np.concatenate((x_train, x_augument_01, x_augument_02))
y_train = np.concatenate((y_train, y_augument_01, y_augument_02))

print(x_train.shape)
print(y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train , train_size=0.8, shuffle=True, random_state=9)

print(x_train.shape) # (9648, 150, 150, 3)
print(y_train.shape) # (9648, 10)
print(x_test.shape)  # (2412, 150, 150, 3)
print(y_test.shape)  # (2412, 10)

np.save('d:/study_data/_data/Project_M/test/_npy/train_x.npy', arr =x_train)
np.save('d:/study_data/_data/Project_M/test/_npy/train_y.npy', arr =y_train)

np.save('d:/study_data/_data/Project_M/test/_npy/test_x.npy', arr =x_test)
np.save('d:/study_data/_data/Project_M/test/_npy/test_y.npy', arr =y_test)
print('End')

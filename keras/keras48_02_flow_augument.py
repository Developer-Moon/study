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

print(x_train.shape) # (60000, 28, 28)


####################################################################################################################################
# 증폭용 데이터

augument_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augument_size)   # np.random - 랜덤하게 정수값을 넣는다
#                            ㄴ(60000, 28, 28)                      # 해석 : 60000개 중에 40000개를 랜덤하게 정수를 뽑는다
                      
                         
# print(x_train.shape)                   # (60000, 28, 28)
print(x_train.shape[0])                  # 60000
print(randidx)                           # [57655 21229 32293 ... 38962 49663 49072]
print(np.min(randidx), np.max(randidx))  # randidx가 랜덤으로 뽑는거라 매번 min, max의 값이 다르다
print(type(randidx))                     # <class 'numpy.ndarray'>


print(x_train.shape) # (60000, 28, 28) 하단 확인 용도
print(y_train.shape) # (60000,)        하단 확인 용도

x_augumented = x_train[randidx].copy()  # .copy() 실 데이터에 영향을 주지 않기 위해 copy() 사용[안전성]
y_augumented = y_train[randidx].copy()

print(x_augumented.shape) # (40000, 28, 28) 상단 shape 확인
print(y_augumented.shape) # (40000,)        상단 shape 확인

####################################################################################################################################
#원본

x_train = x_train.reshape(60000, 28, 28, 1)

print(x_test.shape) # (10000, 28, 28)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) # 4차원 변경
              
x_augumented = x_augumented.reshape(x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 1)  # 4차원 변경

x_augumented = train_datagen.flow(x_augumented, # x
                                  y_augumented, # y
                                  batch_size=augument_size,
                                  shuffle=False, # 여기선 셔플이 필요없다 상단에서 randidx를 사용해서
                                  ).next()[0]    # x의 값을 뽑기위해 [0]

# print(x_augumented)
print(x_augumented.shape)  # (40000, 28, 28, 1)
print(y_augumented.shape)  # (40000,)
print(x_train.shape)       # (60000, 28, 28, 1)
print(y_train.shape)       # (60000,)

x_train = np.concatenate((x_train, x_augumented)) # concatenate 엮다 
y_train = np.concatenate((y_train, y_augumented)) 

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)


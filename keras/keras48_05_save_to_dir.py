from tkinter import Image
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import axis
import numpy as np  

#이미지 4만개 증폭

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

# 40000개의 카피본을 만들기 위해 이러한 작업을 한다(중복되지 않게)
augument_size = 20 # 증가시키다








randidx = np.random.randint(x_train.shape[0], size=augument_size)       # p.random - 랜덤하게 정수값을 넣는다
                                                                        # 60000개 중에 40000개를 랜덤하게 정수를 뽑는다
                      
                         
print(x_train.shape)    # (60000, 28, 28)
print(x_train.shape[0]) # 60000
print(randidx)          # [57655 21229 32293 ... 38962 49663 49072]
print(np.min(randidx), np.max(randidx))
print(type(randidx))    # <class 'numpy.ndarray'>


x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()
print(x_augumented.shape) # (40000, 28, 28)
print(y_augumented.shape) # (40000,)



#원본
x_train = x_train.reshape(60000, 28, 28, 1)
print(x_test.shape) # (10000, 28, 28)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)        #x_train 이랑 똑같이??
                        #   10000      ,      28                  28
                        
x_augumented = x_augumented.reshape(x_augumented.shape[0],       # 4차원 변경
                                    x_augumented.shape[1],    
                                    x_augumented.shape[2], 1
)



import time
start_time = time.time()
print('시작 !!!')

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False,               # 여기선 필요없다? 이미 위에서 랜덤하게 뽑아서?
                                  save_to_dir='d:/study_data/_temp/',
                                  ).next()[0]   # [0] 들어가는게 x값만 뽑을라고   그라면 y는 왜 넣는데?   

end_time = time.time() - start_time
print('걸린시간 :', round(end_time, 3), "초")

# print(x_augumented)
print(x_augumented.shape)  # (40000, 28, 28, 1)




x_train = np.concatenate((x_train, x_augumented))           # concatenate 엮다   클래스 공부해서 괄호 개수에 알아봐라
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape) # (100000, 28, 28, 1) (100000,)




# [실습]
# 1. x_augumented 10개와 x_train 10개를 비교하는 이미지 출력할 것 



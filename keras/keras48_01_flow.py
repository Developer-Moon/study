from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np  

# 1. ImageDataGenerator를 정의
# 2. 파일에서 가져오려면 -> flow_from_directory() // x,y가 튜플 형태로 뭉쳐있다.
# 3. 데이터에서 가져오려면 -> flow() // x,y가 나눠져있다.

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./225,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.01,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)



augument_size = 100 

print(x_train.shape)    # (60000, 28, 28)
print(x_train[0])
print(x_train[0].shape) # (28, 28)
print(x_train[0].reshape(28*28).shape) # (784,)
print(np.tile(x_train[0].reshape(28*28), augument_size).shape)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1).shape) # (100, 28, 28, 1)
  
print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) # (100,)


# x와 y를 여기서 분리한다
x_data = train_datagen.flow(        
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1, 28, 28, 1),   # x
    np.zeros(augument_size),                                                    # y
    batch_size=augument_size,
    shuffle=True,            
).next()



#####     next() 사용 X     #####                                           
print(x_data) 
print(x_data[0])                # 배치만큼 나온다
print(x_data[0][0].shape)       # (100, 28, 28, 1)
print(x_data[0][1].shape)       # (100,)
print(x_data[0][0][0].shape)    # (28, 28, 1)
print(x_data[0][0][0][2].shape) # (28, 1)


 #####     next() 사용 O     #####

print(x_data) 
print(x_data[0])                
print(x_data[0][0].shape)       # (28, 28, 1)
print(x_data[0][1].shape)       # (28, 28, 1)
print(x_data[0][0][0].shape)    # (28, 1)
print(x_data[0][0][0][2].shape) # (1,)



import matplotlib.pylab as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7, 7, i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')  
    # plt.imshow(x_data[0][0][i], cmap='gray')  # next() 비사용시
plt.show()    

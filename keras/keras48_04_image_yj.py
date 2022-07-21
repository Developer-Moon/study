from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(   # ImageDataGenerator 이미지를 숫자화(수치화)
    rescale=1./255,   # 나는 1개의 이미지를 픽셀 255개로 나눌거야 스케일링한 데이터로 가져오겠다 ..? 
    horizontal_flip=True,
    # vertical_flip=True,  # 반전시키겠냐 ? / true 네! 라는 뜻이라고함
    width_shift_range=0.1, # 가로 세로
    height_shift_range=0.1, # 상 하
    rotation_range=5, # 돌리겟다?
    zoom_range= 0.1, # 확대
    # shear_range= 0.7, # 선생님 : 알아서 찾아 ~ ;;;  /  선생님 : 찌글찌글 ?? ;
    fill_mode='nearest'  
)

augument_size = 40000   # 4만장을 늘리겠다 ~ 라는 거임   /  이 데이터는 아래에서 60000의 데이터중 랜덤하게 정수를 뽑을 예정임 ~ 
randidx = np.random.randint(x_train.shape[0], size=augument_size)    # https://www.sharpsightlabs.com/blog/np-random-randint/  [np.random.randint 설명링크]
                             # 60000 - 40000 /  60000개에서 40000만개의 데이터를  np.random.randint을 사용해서 정수를 랜덤으로 뽑아서 randidx안에 넣겠다는 뜻.
 # np.random.randint : 랜덤하게 정수를 넣는다.
print(x_train.shape[0])  # 60000, 28, 28, 1 중에서 60000을 출력해주고 [1]일 경우는 28을 출력해준다.
print(randidx.shape)  # (40000,)
print(np.min(randidx), np.max(randidx))   #  최소값--> 2 59996  <-- 최대값
print(type(randidx))   # <class 'numpy.ndarray'>   numpy형태는 기본적으로 리스트 형태이다.

x_augumented = x_train[randidx].copy()   # 연산할 때 새로운 공간을 만들어서 할때는 .copy()를 사용하면 새로운 메모리을 확보해서 그 공간에서 작업을 하겠다는 뜻이다. 
                                              # 즉 ! 원본을 전혀~~ 안건들고 새로운 공간에서 연산을 하겠다는 뜻! / 이것으로 인해서 안전성이 올라갔다
# x_train에서 randidx를 뽑아서 x_augumented에다가 저장하겟다는 뜻
y_augumented = y_train[randidx].copy()   
# y_train에서 randidx를 뽑아서 y_augumented에다가 저장하겟다는 뜻
 
print(x_augumented.shape)    # (40000, 28, 28)
print(y_augumented.shape)    # (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

x_augumented = x_augumented.reshape(x_augumented.shape[0],
                                    x_augumented.shape[1],
                                    x_augumented.shape[2], 1
                                    )
 
x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0] 

print(x_augumented)
print(x_augumented.shape)    # (40000, 28, 28, 1)
                            
x_train = np.concatenate((x_train, x_augumented))  #엮다   클래스 공부해라 ~~ 
y_train = np.concatenate((y_train, y_augumented))  
                    
print(x_train.shape, y_train.shape)      # (100000, 28, 28, 1) (100000,)             
 

# [실습]
# 1. x_augumented 10개와 x_train  10개를 비교하는 이미지를 출력할 것.


import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))       # 2x10 크기로 잘라라
for i in range(20):              # 0 ~ 19 범위
    if i <= 9:                   # 0 ~ 9 
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.imshow(x_augumented[i], cmap='gray')
    else:                        # 10 ~ 19 
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.imshow(x_train[randidx[i-10]], cmap='gray')
        
plt.show()


# import matplotlib.pyplot as plt

# plt.figure(figsize = (9,6))     # figsize(가로길이,세로길이)
# plt.plot(hist.history['loss'], marker='.', color='red', label='loss')           # label='loss' 해당 선 이름
# plt.plot(hist.history['val_loss'], marker='.', color='blue', label='val_loss')  # marker='.' 점으로 찍겟다
# plt.grid()                        # plt.grid(True)    # grid: 그리다
# plt.title('asaql')                # title의 이름을 asaql로 하겠다
# plt.ylabel('loss')                # y라벨의 이름을 loss로 하겠다
# plt.xlabel('epochs')              # x라벨의 이름을 epochs로 하겠다
# plt.legend(loc = 'upper right')   # 그래프가 없는쪽에 알아서 해준다 굳이 명시할 필요 없음
# # plt.legend()   # 그래프가 없는쪽에 알아서 해준다 굳이 명시를 안 할 경우 사용법
# plt.show()    # 그래프를 보여줘라
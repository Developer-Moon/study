from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten        
"""
# DNN모델
model = Sequential()                             
model.add(Dense(units=10, input_shape=(3, )))   # (batch_size, input_dim) 의 형태    input_shape=(10, 10, 3)
model.summary()
# (input_dim + bias) * units = summary param 개수 (Dense모델 - DNN)
"""



# CNN모델                                                                                         
model = Sequential()                                                                              
model.add(Conv2D(filters=10, kernel_size=(4,4), input_shape=(8, 8, 1)))                              # (5, 5, 10)   param : 170                                                                   
model.add(Conv2D(7, (2, 2), activation='relu'))  # filters 7   kernel_size (2, 2)                    # (4, 4, 7)    param : 287
model.add(Conv2D(5, (2, 2), activation='relu'))  # filters 7   kernel_size (2, 2)                  
# model.add(Flatten()) # (N, 252)                                                                           
# model.add(Dense(32, activation='relu'))                                                                                                         
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))
model.summary() 



# filters : node의 개수 [10을 곱해주겠다 output이 된다?]
# kernel_size=(2,2) : 이미지를 자르는 규격 2x2사이즈로 이미지를 자른다
# input_shape=(5, 5, 1) : 5 x 5짜리 1 흑백이미지          컬러면 3          (batch_size, rows, columns, channels)

# 4차원을 2차원으로 바꿀때 (N, 4, 3, 2) -> (N, 4x3x2)  데이터를 펼친다

# outputshape의 row, columns수 = (input_shape - kernal_size) + 1
# CNN모델 output shape의 형태는 (batch_size, row, columns, channls)로 출력됨
# Dense모델 param 갯수 = (input_dim x units) + bias node(nuit)
# CNN모델 param 갯수 = (kernel_size x channels x filters) + bias node(filter) - 죄송합니다 ㅜㅜ
                          # 16          1         10              10
                         







 
#  none는 입력될 이미지의 개수는 정해지지 않아서 none이며 batch size가 입력됨   batch size - 행의개수(곱해지는 수)
#  4, 4 : MNIST 테이터는 픽셀
#  10 : 10은 색 채널을 의미 MNIST데이터는 회색조(grayscale) 이미지 이므로 한 개의 채널을 가짐
#  kernel_size는 합성곱에 사용되는 필터 (커널)의 크기





 
"""                             (batch_size, input_dim)

model.add(Dense(10, activation='relu', input_dim=8))이라면

tf.keras.layers.Dense(
    units,                                 아웃풋 노드의 개수 10   units=10 이라고 써도 가능
    activation=None,
    use_bias=True,                          bias를 쓸꺼냐  쓴다
    kernel_initializer="glorot_uniform",     
    bias_initializer="zeros",
    kernel_regularizer=None,            
    bias_regularizer=None,              
    activity_regularizer=None,             
    kernel_constraint=None,             
    bias_constraint=None,               
    **kwargs 
)

"""







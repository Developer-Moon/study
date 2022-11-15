from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D      
"""
# DNN모델
model = Sequential()                             
model.add(Dense(units=10, input_shape=(3, )))   # (batch_size, input_dim) 의 형태    input_shape=(10, 10, 3)
model.summary()
# (input_dim + bias) * units = summary param 개수 (Dense모델 - DNN)
"""



# CNN모델                                                                                         
model = Sequential()                                                                             
model.add(Conv2D(filters=64, kernel_size=(3,3), # Conv2D 복잡한
                 padding='same',           # padding 0이라는 패딩을 씌워서 이미지를 조각낼때 가장자리 부분을 두번 이상 넣어줘서 다른 부분보다 덜 학습되는걸 방지 
                 input_shape=(28, 28, 1)))  # 통상 shape를 다음 레이어에도 유지하고 싶을때 padding을 쓴다                                                                                                         
model.add(MaxPooling2D())    #(14, 14, 64) 
model.add(Conv2D(32, (2, 2),
                 padding='valid',          # 디폴트
                 activation='relu'))                          
model.add(Flatten()) # (N, 252)                                                                         
model.add(Dense(32, activation='relu'))                                                                                                         
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary() 


# 이미지에서 제일 큰값을 뺀다 Maxooling








 
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







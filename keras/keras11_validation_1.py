from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
import numpy as np  


#1. 데이터                        
x_train = np.array(range(1,11))   
y_train = np.array(range(1,11))
x_test = np.array([11,12,13])     # test셋은 evaluate, predict에서 사용   
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])      # val 검증의 약자 = validation(검증)   
y_val = np.array([14,15,16])               
# train : 공부(훈련)  validation : 모의고사  test : 수능         
                                   


#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=1))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # 훈련하고 문제풀고 훈련하고 문제풀고



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                                 # 여기서 수능보고 
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)


"""
Epoch 100/100
10/10 [==============================] - 0s 4ms/step - loss: 5.5280e-13 - val_loss: 2.4253e-12  <---- 여기에 validation이 생긴다  검증을 했을때 로스보다 값이 더 좋지 않아야한다 
1/1 [==============================] - 0s 63ms/step - loss: 1.8190e-12                                과적합 구간인 경우 로스가 더 좋지 않을 수 있다
loss :  1.8189894035458565e-12
17의 예측값 :  [[16.999998]]

                                            Training dataset     Validation dataset     Test dataset
-------------------------------------------------------------------------------------------------------
학습 과정에서 참조할 수 있는가?                       O                     O                   X
모델의 인자값(가중치) 설정에 이용되는가?               O                     X                   X
모델의 성능 평가에 이용 되는가?                       X                     O                   O
"""


# val_loss : loss값보다 떨어져야 한다  val_loss를 이용하여 overfitting(과적합) 방지
# 우리의 목적은 학습을 통해 머신 러닝 모델의 underfitting된 부분을 제거해나가면서 overfitting이 발생하기 직전에 학습을 멈추는 것
# 이를 위해 머신 러닝에서는 validation set을 이용

# 머신 러닝의 궁극적인 목표는 training dataset을 이용하여 학습한 모델을 가지고 test dataset를 예측하는 것
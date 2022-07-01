from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense  
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score 
import numpy as np 
import time

#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=13))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))


start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                                                            
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.2)  # hist = history 

                                                                

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) # loss :  49.679412841796875    val_loss: 65.4229

print("===================================================================================================")
# print(hist) # <tensorflow.python.keras.callbacks.History object at 0x000001D47242FE80>
print("===================================================================================================")
# print(hist.history)
# {'loss': [297.57769775390625, 93.09060668945312, 125.23074340820312, 81.37921142578125, 87.60233306884766, 76.19335174560547, 77.59649658203125, 80.06200408935547,
#           72.04061889648438, 70.94953155517578, 69.230712890625],
# 'val_loss': [90.7829360961914, 587.2574462890625, 141.90908813476562, 126.72649383544922, 85.05255126953125,
#           75.07796478271484, 184.0257110595703, 69.66654205322266, 63.8183708190918, 67.80097961425781, 66.54901123046875]}
# hist.history - t, value류 형태로 로스와 발로스 형태를 반환
print("===================================================================================================")
# 두개중 1개만 보고싶을때 hist.history [loss] or  hist.history [val]
# print(hist.history['loss'])
print("===================================================================================================")
print(hist.history['val_loss'])
print("===================================================================================================")


end_time = time.time() - start_time
print('걸린시간 :', end_time)


import matplotlib.pyplot as plt    
plt.figure(figsize=(9,6))                                                   # 판 크기
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')           # marker=로스부분 .으로 표시      c='red' 그래프를 붉은컬러로     label='loss' 이그래프의 이름(label)은 loss
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 



plt.grid()           # 모눈종이로 만들자
plt.title('안결바보') # 한글 깨진걸 찾아서 넣어라
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.legend(loc='upper right') #   label값이 레전드에 명시가 되며 이걸 우측상단에 올린다 location = loc            위치값 upper right', 'lower left', 'center left', 'center 이런게 있다
plt.legend() # 자동으로 빈 공가넹 표시
plt.show()


# 로스의 최소값에서 최적의 웨이트를 찾는다 : 그래프에서 최적의 가중치를 찾아 세이브 한다

# loss :  32.253082275390625   val_loss: 48.4251

# 로스와 발로스의 간격이 좁은게 좋다

# 성능만 따지면 로스와 발로스의 간격이 넓어도 로스값이 적은게 좋다(로스를 신뢰하는게 좋다) [세팅을 두개 했을때]


# 시각화를 잘 해야한다(보고서를 제출할때도)


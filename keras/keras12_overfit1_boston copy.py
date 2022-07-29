from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score 
from sklearn.datasets import load_boston  
import time


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      
                  
print(x.shape, y.shape)   # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)



#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=13))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일, 훈련
start_time = time.time()
model.compile(loss='mse', optimizer='adam')                   
hist = model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
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
print('time : ', end_time)

import matplotlib.pyplot as plt    
plt.figure(figsize=(6,6))                                                   # 팝업창이 뜰 때의 사이즈
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')           # marker= 수치부분 .으로 표시 c='red' 그래프를 붉은컬러로  label='loss'
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 



plt.grid()             # 모눈종이로 만들자
plt.title('안결바보')   
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.legend(loc='upper right') #   label값이 레전드에 명시가 되며 이걸 우측상단에 올린다 location = loc    위치값 upper right', 'lower left', 'center left', 'center 이런게 있다
plt.legend()           # label의 이름들을 빈 공간에 알아서 넣어라
plt.show()



# 로스와 발로스의 간격이 좁은게 좋다

# 성능만 따지면 로스와 발로스의 간격이 넓어도 로스값이 적은게 좋다(로스를 신뢰하는게 좋다) [세팅을 두개 했을때]

# 시각화를 잘 해야한다(보고서를 제출할때도)



"""
loss : 21.61882972717285 ----- Normal
r2 : 0.738325009535564

loss : 20.06865882873535 ----- validation_split
r2 : 0.7570883292496796
"""



from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense  
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_diabetes     
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
datasets = load_diabetes()                      
x = datasets.data
y = datasets.target 

print(x) 
print(y) 
print(x.shape, y.shape)         #(442, 10) (442,)     컬럼 = 10    스칼라 = 422

print(datasets.feature_names)   #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR) 

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9,
    shuffle=False
    )



#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

import time
start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                   
hist = model.fit(x_train, y_train, epochs=500, batch_size=1000, verbose=1, validation_split=0.2)


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss) 
end_time = time.time() - start_time
print('걸린시간 :', end_time)


import matplotlib.pyplot as plt    
plt.figure(figsize=(9,6))                                                   # 판 크기
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')           # marker=로스부분 .으로 표시      c='red' 그래프를 붉은컬러로     label='loss' 이그래프의 이름(label)은 loss
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 


plt.grid()            #모눈종이로 만들자
plt.title('안결바보')  #한글 깨진걸 찾아서 넣어라
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.legend(loc='upper right') #   label값이 레전드에 명시가 되며 이걸 우측상단에 올린다 location = loc 
plt.legend() # 자동으로 빈 공가넹 표시
plt.show()


# loss : 1710.50732421875    val_loss: 3489.7583

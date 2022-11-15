from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense  
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score 
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

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

scaler = MinMaxScaler()     
# scaler = StandardScaler() 
# scaler = MaxAbsScaler()                                                                                  
# scaler = RobustScaler()
scaler.fit(x_train)                    
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)  


#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=13))
model.add(Dense(32, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.summary()


# model.save('./_save/keras23_1_save_model.h5') # 이걸 불러오면 이 모델을 불러온다









start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')             
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)                                                   
hist = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1, callbacks=[earlyStopping], validation_split=0.2)


model.save('./_save/keras23_3_save_model.h5') # 이걸 불러오면 이 모델을 불러온다                                                               

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) # loss :  49.679412841796875    val_loss: 65.4229


y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)



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



# loss    : 11.616869926452637
# r2스코어 : 0.8610138651001857
# 걸린시간 : 5.067552804946899

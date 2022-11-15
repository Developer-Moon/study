from sklearn.datasets import load_boston   
from sklearn.model_selection import train_test_split   
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input                    
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler            # 전처리 = preprocessing  # 이상치를 잘 고르는 애가 있다. 
import numpy as np       
from sklearn import metrics         
from sklearn.metrics import r2_score 

           
     
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
    random_state=66
    )

                                                                              
# scaler = MinMaxScaler() 
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)                    
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)       



#2. 모델구성
input_01 = Input(shape=(13,))
dense_01 = Dense(14)(input_01)
dense_02 = Dense(20, activation='relu')(dense_01)
dense_03 = Dense(30, activation='relu')(dense_02)
dense_04 = Dense(20, activation='relu')(dense_03)
dense_05 = Dense(10, activation='relu')(dense_04)
output_01 = Dense(1)(dense_05)
model = Model(inputs=input_01, outputs=output_01)

""" 기존 Sequential모델
model = Sequential()
model.add(Dense(14, input_dim=13))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
"""
import time
start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])  # True 또는 False, 양성 또는 음성 등 2개의 클래스를 분류할 수 있는 분류기를 의미                                                                                                                                                                                                                                                                           
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  
# callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score # 두개 같이 쓸 수 있다 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
end_time = time.time() - start_time
print('걸린시간 :', end_time)


import matplotlib.pyplot as plt    
plt.figure(figsize=(9,6))                                                   
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')           
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 


plt.grid()            
plt.title('안결바보') 
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend() 
plt.show()




"""
loss: 13.3111
val_loss : 35.0299
mae : 2.6359
val_mae : 3.7203
r2스코어 : 0.8407432260663265
####################################
[scaler = MinMaxScaler]

loss     : 17.2238
r2스코어  : 0.7939315648550948
####################################
[scaler = StandardScaler]                       

loss     : 0.8640
r2스코어  : 0.8598465102174764
####################################
[scaler = SMaxAbsScaler]                     함수모델 사용 

걸린시간 : 38.587186336517334                 걸린시간 : 36.59080147743225
loss    : 8.348922729492188                  loss    : 10.792678833007812          
r2스코어 : 0.9001121023160439                 r2스코어 : 0.8624553231648865
####################################
[scaler = RobustScaler]

loss    " 13.79703426361084
r2스코어 : 0.8349300368196414

"""





































from sklearn.datasets import fetch_california_housing  
import numpy as np
from sklearn import metrics         
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler    
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score 
import time


#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #


#1. 데이터 
datasets = fetch_california_housing()
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

print(np.min(x_train))  
print(np.max(x_train))  
print(np.min(x_test)) 
print(np.max(x_test))  



#2. 모델구성
model = Sequential()
model.add(Dense(50, input_dim=8))
model.add(Dense(70, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary() # Total params: 15,591


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                              
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)



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


# 시각화를 잘 해야한다(보고서를 제출할때도)
# loss :  0.6256002187728882
# val_loss: 0.5744
# r2스코어 : 0.5512402837274273


####################################
# 20220703

# loss: loss : 0.4931
# val_loss : 0.4429
# mae : 0.5236
# val_mae : 0.5020
# r2스코어 : 0.6514661215114226
####################################

"""""""""""""""""""""""""""""""""
[scaler = MinMaxScaler]

loss     : 0.2817
val_loss : 0.2567
mae      : 0.3486
val_mae  : 0.3382
r2스코어 : 0.7979576497705312
"""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""
[scaler = StandardScaler]        <<<<<<<<<<< Fantastic

loss     : 0.2064
val_loss : 0.2372
mae      : 0.3123
val_mae  : 0.3234
r2스코어 : 0.8105301815995982

[scaler = MaxAbsScaler]

loss     : 0.4123687744140625
r2스코어 : 0.7041968935115241
"""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""
[scaler = RobustScaler]

loss     : 0.2834553122520447
r2스코어 : 0.7966699568426323
"""""""""""""""""""""""""""""""""

















             




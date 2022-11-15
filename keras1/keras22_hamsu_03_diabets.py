import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_diabetes  
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input      
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 

   
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


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )

# scaler = MinMaxScaler() 
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)     # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)       # train이 변한 범위에 맞춰서 변환됨

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위



#2. 모델구성

input_01 = Input(shape=(10,)) 
dense_01 = Dense(10)(input_01)
dense_02 = Dense(100)(dense_01)
dense_03 = Dense(100)(dense_02)
dense_04 = Dense(100)(dense_03)
dense_05 = Dense(100)(dense_04)
dense_06 = Dense(100)(dense_05)
dense_07 = Dense(10)(dense_06)
output_01 = Dense(1)(dense_07)
model = Model(inputs=input_01, outputs=output_01)
model.summary() # Total params: 42,631

""" 기존 Sequential모델
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(100))  
model.add(Dense(100)) 
model.add(Dense(100))  
model.add(Dense(100))  
model.add(Dense(100))  
model.add(Dense(10))  
model.add(Dense(1))
model.summary() 
"""



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# 메트릭스를 쓰면 로스가 두개가 나온다   앞에껀 binary_crossentropy 뒤에껀 에큐러시(정확도)
# loss :  [0.2551944851875305, 0.9122806787490845]   

# 이진분류에서는 mse를 신뢰할 수 없다








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


# loss :  3197.76123046875
# val_loss: 3192.9006
# r2스코어 : 0.5072817321713152

####################################
# 20220703

# loss : 3807.1777
# val_loss : 3079.1750 
# mae : 48.9231
# val_mae : 45.8657
# r2스코어 : 0.41338157521308094
####################################

"""""""""""""""""""""""""""""""""
[scaler = MinMaxScaler]

loss     : 5143.45166015625
val_loss : 48.58793640136719
mae      : 1.0021
val_mae  : 2.9203
r2스코어 : 0.48744652984284853
"""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""
[scaler = StandardScaler]           

loss     : 3343.350830078125
val_loss : 47.08863067626953
mae      : 0.6448
val_mae  : 3.0111
r2스코어 : 0.49632361216845167


[scaler = MaxAbsScaler]

loss     : 3423.301025390625
r2스코어 : 0.4725301075221435

                                         함수모델 사용 
[sclaer = RobustScaler]

loss     : 3281.2236328125              loss : 3754.879638671875
r2스코어 : 0.49442168387100893           r2스코어 : 0.42143977370513974
"""""""""""""""""""""""""""""""""




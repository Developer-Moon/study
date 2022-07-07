from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.datasets import load_wine
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

import tensorflow as tf            
tf.random.set_seed(66)


#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target 
print(x.shape) # (178, 13)
print(y.shape) # (178,)
# print(np.unique(x))
print(np.unique(y, return_counts=True))     # [0 1 2] - (array([0, 1, 2]), array([59, 71, 48], dtype=int64)) 0이 59개  1이 71개  2가 48개

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)
print(y)
print(y.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )





scaler = MinMaxScaler() 
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)     # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)        # train이 변한 범위에 맞춰서 변환됨

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위



#2. 모델구성
input_01 = Input(shape=(13,))
dense_01 = Dense(100)(input_01)
dense_02 = Dense(100)(dense_01)
dense_03 = Dense(100)(dense_02)
dense_04 = Dense(100)(dense_03)
dense_05 = Dense(100)(dense_04)
dense_06 = Dense(100)(dense_05)
dense_07 = Dense(100)(dense_06)
dense_08 = Dense(100)(dense_07)
dense_09 = Dense(100)(dense_08)
dense_10 = Dense(100)(dense_09)
dense_11 = Dense(100)(dense_10)
output_01 = Dense(3, activation='softmax')(dense_11)
model = Model(inputs=input_01, outputs=output_01)



"""
기존 Sequential모델
model = Sequential()
model.add(Dense(100, input_dim=13))  
model.add(Dense(100))           
model.add(Dense(100))           
model.add(Dense(100))           
model.add(Dense(100))           
model.add(Dense(100))           
model.add(Dense(100))           
model.add(Dense(100))                 
model.add(Dense(100))           
model.add(Dense(100))           
model.add(Dense(100))           
model.add(Dense(3, activation='softmax'))
"""
                           
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',            # 다중분류의 loss는 $$$당!분!간$$$ categorical_crossentropy 만 쓴다 (20220704)
              optimizer='adam',
              metrics=['accuracy'])
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, callbacks=[earlyStopping], verbose=1) 


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 

from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = np.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)





import matplotlib.pyplot as plt    
plt.figure(figsize=(9,6))                                                   
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')           
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 



plt.grid()      
plt.title('안결바보')
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.legend(loc='upper right') #   label값이 레전드에 명시가 되며 이걸 우측상단에 올린다 location = loc            위치값 upper right', 'lower left', 'center left', 'center 이런게 있다
plt.legend()
plt.show()



"""""""""""""""""""""""""""""""""
[scaler = MinMaxScaler]

loss     :  0.09179487079381943      함수모델 사용 
val_loss :
mae      : 
val_mae  :                           loss :  0.09179487079381943
acc스코어 : 0.9491525423728814       acc스코어 : 0.9491525423728814
"""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""
[scaler = StandardScaler]           

loss     : 0.21569064259529114
mae      :
val_mae  :
acc스코어 : 0.9152542372881356




[scaler = MaxAbsScaler]
loss     :  2.3755462169647217
acc스코어 : 0.7627118644067796




[scaler = RobustScaler]

loss     :  0.25593098998069763
acc스코어 : 0.8983050847457628
"""""""""""""""""""""""""""""""""


















from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input 
from sklearn.model_selection import train_test_split        
from sklearn.datasets import load_wine, load_digits                             
from sklearn.metrics import r2_score
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

import tensorflow as tf            
tf.random.set_seed(66)         #텐서플로의 난수표 66번 사용 이 난수는 w = 웨이트


#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (1797, 64) (1797,)    8x8(64)이미지가 1797개 있다는 말  input_dim = 64
print(np.unique(y, return_counts=True))      # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))    이걸 1797,10으로 변환

""" 이미지 보는법 
import matplotlib.pyplot as plt    
plt.gray()
plt.matshow(datasets.images[3])
plt.show()
"""


from tensorflow.keras.utils import to_categorical # 범주
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
x_test = scaler.transform(x_test)       # train이 변한 범위에 맞춰서 변환됨

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위

#2. 모델구성
input_01 = Input(shape=(64,))
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
output_01 = Dense(10, activation='softmax')(dense_01)
model = Model(inputs=input_01, outputs=output_01)
model.summary()

"""기존 Sequential모델
model = Sequential()
model.add(Dense(100, input_dim=64))  
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
model.add(Dense(10, activation='softmax')) 
"""


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',            
              optimizer='adam',
              metrics=['accuracy'])
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  # batch_size 디폴트값 32



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



# r2 = r2_score(y_test, y_predict)
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

loss     :  0.3825773000717163        함수모델 사용 
val_loss :
mae      : 
val_mae  :                             loss :  0.11701111495494843
acc스코어 : 0.8888888888888888        acc스코어 : 0.9646464646464646
"""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""
[scaler = StandardScaler]           

loss     : 0.4895826280117035
mae      :
val_mae  :
acc스코어 : 0.8653198653198653


[scaler = RobustScaler]

loss    :  0.639191210269928
cc스코어 : 0.7811447811447811


[scaler = MaxAbsScaler]

loss      :  0.3825773000717163
acc스코어 : 0.8888888888888888
"""""""""""""""""""""""""""""""""












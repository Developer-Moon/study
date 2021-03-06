from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split        
from sklearn.datasets import fetch_covtype                            
from sklearn.metrics import r2_score
import numpy as np 
import pandas as pd 
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('쥐피유 돌아유')
    aaa = 'gpu - 쥐피유 돌아유'
else:
    print('내가 돌아유')
    aaa = 'cpu - 내가 돌아유'




#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

import tensorflow as tf            
tf.random.set_seed(66)         #텐서플로의 난수표 66번 사용 이 난수는 w = 웨이트

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y, return_counts=True))     # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
                                            # 4가 잘 안나온다? 4를 삭제할까? 라는 생각을 해야한다??
                                            # 이 데이터를 증폭시킨다?                     
                                            # !!!!!!!!!!!!!!!!! - 여기서 7개의 컬럼인데 
y = pd.get_dummies(y)
print(pd.get_dummies(y))
print(y.shape)









"""
from tensorflow.keras.utils import to_categorical # 범주
y = to_categorical(y)
print(y)
print(y.shape) # (581012, 8)                  !!!!!!!!!!!!!!!!! - 갑자기 여기서는 왜 8개가 나오나??
"""

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
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
model = Sequential()
model.add(Dense(500, input_dim=54))  
model.add(Dense(400))                 
model.add(Dense(300))           
model.add(Dense(200))           
model.add(Dense(100))           
model.add(Dense(7, activation='softmax')) 

start_time = time.time()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',            # 다중분류의 loss는 $$$당!분!간$$$ categorical_crossentropy 만 쓴다 (20220704)
              optimizer='adam',
              metrics=['accuracy'])
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2, callbacks=[earlyStopping], verbose=1) 


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 


from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)

y_test = tf.argmax(y_test, axis=1)
print(y_test)


end_time = time.time() - start_time

# r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)  
print(aaa, 'acc스코어 :', acc)
print(end_time)


###################################
# loss :  1.1296087503433228
# accuracy :  0.5815765857696533
# acc스코어 : 0.5815765591913797
###################################
"""""""""""""""""""""""""""""""""
[scaler = MinMaxScaler]

loss     :  0.6416857838630676      <<<<<<<<<<< Fantastic
val_loss :
mae      : 
val_mae  : 
acc스코어 : 0.7219116067051228
"""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""
[scaler = StandardScaler]           

loss     : 0.6408393979072571
mae      :
val_mae  :
acc스코어 : 0.7222193246894134



[scaler = MaxAbsScaler]

loss      : 0.6470345258712769
acc스코어 : 0.7148080152711569


[scaler = RobustScaler]

loss      : 0.6389427185058594
acc스코어 : 0.722855622894218
"""""""""""""""""""""""""""""""""





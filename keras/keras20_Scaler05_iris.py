from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split        
from sklearn.datasets import load_iris                              
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
datasets = load_iris()
print(datasets.DESCR)
# Number of Instances: 150 (50 in each of three classes) 50개중 3개의 클래스가 있다
# Number of Attributes: 4 numeric 컬럼이 4개 (피처, 열)
# class: - Iris-Setosa - Iris-Versicolour - Iris-Virginica    ->    y값은 3개

print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x)
print(y)

print(x.shape, y.shape)        # (150, 4) (150,)   
print('y의 라벨값 :', np.unique(y))   # y의 라벨값 : [0 1 2] - 판다스에서 있다 : y값을 (150, 3) 으로 바꿔준다

from tensorflow.keras.utils import to_categorical # 범주
y = to_categorical(y)
print(y)
print(y.shape) # (150, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )


# scaler = MinMaxScaler() 
# scaler = StandardScaler()
scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)     # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)       # train이 변한 범위에 맞춰서 변환됨

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위




#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=4))  
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
model.add(Dense(3, activation='softmax')) # 다중분류 activation='softmax에서 y값이 3개 즉  1번 2번 3번 의 꽃이 있으니까 마지막 아웃풋이 3개가 된다

                                          # 여기까지 x-shape와 y-shape가 다르다 (150, 4), (150, 1) 여기서 output이 3이니까 y shape를 바꿔줘야한다 (150, 3)으로

                                          # y값의 라벨의 개수는 3개
                                          # 소프트맥스의 연산의 합은 1    
                                          # y에 찾아야하는 개수가 3개일때 
                                         

import time
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',            # 다중분류의 loss는 $$$당!분!간$$$ categorical_crossentropy 만 쓴다 (20220704)
              optimizer='adam',
              metrics=['accuracy'])
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[earlyStopping], verbose=1) 

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss) 
# print('accuracy : ', acc)                  # 밑에와 같음

results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 
"""
print('=============== y_test[:5]===============')
print(y_test)  # 앞에서 부터 5번째 까지       # [1. 0. 0.]... One Hot 상태
print('=============== y_pred ==================')
y_predict = model.predict(x_test)           # [:5]
print(y_predict)                            # [9.99867916e-01 1.32123358e-04 9.05633400e-18] 이러한 값들이 나온다 
y_predict = np.argmax(y_predict, axis=1)    # argmax 첫번째 y_predict값에서 agmax를 사용하여 가장 큰값의 인덱스를 반환 axis = 0 은 각column중 가장 큰 값의 index   axis = 1 은 행
print(y_predict)                            # [0 1 2 2 2 1 1 2 0 0 0 0 2 0 0 1 2 0 0 0 1 2 0 1 1 2 0 2 2 0 1 2 2 2 0 0 0 2 2 1 0 2 1 1 1 0 2 2 2 2] 행당 max값을 반환한 값

print(y_test)                               # [1. 0. 0.]... 이러한 값들이 나온다 y_test와 y_predict의 값을 비교하기 위해선 와꾸를 맞춰줘야 하니까
y_test = np.argmax(y_test, axis=1)          # x_test도 agmax를 사용하여 가장 큰값의 인덱스를 반환
print(y_test)                               # [0 1 2 2 2 1 1 2 0 0 0 0 1 0 0 1 2 0 0 0 1 2 0 1 1 2 0 2 2 0 1 2 2 2 0 0 0 2 2 1 0 2 1 1 1 0 2 2 2 2]
"""

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

loss     : 0.012187564745545387      <<<<<<<<<<< Fantastic
val_loss :
mae      : 
val_mae  : 
acc스코어 : 1.0
"""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""
[scaler = StandardScaler]           

loss     : 0.026125220581889153
mae      :
val_mae  :
acc스코어 : 1.0



[sclaer = MaxAbsScaler]

loss      : 0.03311080485582352
acc스코어 : 1.0




[sclaer = RobustScaler]

loss      : 0.015602652914822102
acc스코어 : 1.0
"""""""""""""""""""""""""""""""""























































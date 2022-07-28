from sklearn.datasets import load_boston   
from sklearn.model_selection import train_test_split   
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense                    
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler            # 전처리 = preprocessing  # 이상치를 잘 고르는 애가 있다. 
import numpy as np            
from tensorflow.python.keras.callbacks import EarlyStopping 
from sklearn.metrics import r2_score, accuracy_score 


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )

# scaler = MinMaxScaler()     0과 1사이로 수렴  식 - x = (x - np.min(x)) / (np.max(x) - np.min(x))          [정규화]
# scaler = StandardScaler()   x - 평균 / 표준편차                                                           [표준화]
# scaler = MaxAbsScaler()                                                                                  
scaler = RobustScaler()
"""
scaler.fit(x_train)                    # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
"""
x_train = scaler.fit_transform(x_train) # 위에꺼 한번에 가능
x_test = scaler.transform(x_test)        # train이 변한 범위에 맞춰서 변환됨

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위


# 하나하나 하려고 하면 
# print(np.min(x))                                # x의 최소값 - 0.0
# print(np.max(x))                                # x의 최대값 - 711.0 
# x = (x - np.min(x)) / (np.max(x) - np.min(x))   # MinMax로 작업하려면
# print(x[:10])                                  




#2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=13))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse',           # 분류 모델중 이진 분류는 무조건 binary_crossentropy를 쓴다 - 보스턴은 회귀모델 이므로 
              optimizer='adam',
              metrics=['mae'])      # metrics=평가지표를 판단   받아들이는게 리스트 형태 ['accuracy', 'mse']        
                                    # True 또는 False, 양성 또는 음성 등 2개의 클래스를 분류할 수 있는 분류기를 의미                                                                                                                                                                                                                                                                           
      
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=10, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  
# callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)1




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
[scaler = SMaxAbsScaler]

loss    : 8.348922729492188
r2스코어 : 0.9001121023160439
####################################
[scaler = RobustScaler]

loss    " 13.79703426361084
r2스코어 : 0.8349300368196414

"""
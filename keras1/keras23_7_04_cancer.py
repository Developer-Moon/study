from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input 
from sklearn.model_selection import train_test_split                                     
from sklearn.datasets import load_breast_cancer  
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler   


#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data           
y = datasets.target 
print(x.shape, y.shape)      

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )
        
scaler = StandardScaler()
scaler.fit(x_train)     # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)       # train이 변한 범위에 맞춰서 변환됨



#2. 모델구성
input_01 = Input(shape=(30,))
dense_01 = Dense(100)(input_01)
dense_02 = Dense(100, activation='relu')(dense_01)
dense_03 = Dense(100)(dense_02)
dense_04 = Dense(100)(dense_03)
dense_05 = Dense(100)(dense_04)
dense_06 = Dense(100)(dense_05)
dense_07 = Dense(100, activation='sigmoid')(dense_06)
dense_08 = Dense(100)(dense_07)
output_01 = Dense(1, activation='sigmoid')(dense_08)
model = Model(inputs=input_01, outputs=output_01)
model.summary()

# model.save('./_save/keras23_7_04_cancer_01_save_model.h5')          
# model.save_weights('./_save/keras23_7_04_cancer_03_save_weights.h5')   
         
# model = load_model('./_save/keras23_7_04_cancer_01_save_model.h5')        
# model.load_weights('./_save/keras23_7_04_cancer_03_save_weights.h5')        
model.load_weights('./_save/keras23_7_04_cancer_04_save_weights.h5')   
                                        
# #3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
# from tensorflow.python.keras.callbacks import EarlyStopping      
# earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
# hist = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1) 

# model.save('./_save/keras23_7_04_cancer_02_save_model.h5') 
# model.save_weights('./_save/keras23_7_04_cancer_04_save_weights.h5')

# model = load_model('./_save/keras23_7_04_cancer_02_save_model.h5')  


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss) 
y_predict = model.predict(x_test)

y_predict = y_predict.flatten()            
y_predict = np.where(y_predict > 0.5, 1 , 0)
print(y_predict)
from sklearn.metrics import r2_score, accuracy_score 
acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)

# loss :  [0.14472554624080658, 0.9736841917037964] --------------------- 기존훈련        
# acc스코어 : 0.9736842105263158

# loss :  [0.09215070307254791, 0.9824561476707458] --------------------- 01_save_model
# acc스코어 : 0.9824561403508771

# loss :  [0.14472554624080658, 0.9736841917037964] --------------------- 02_save_model : 모델과 가중치 같이 저장          [기존 훈련의 가장 좋은 값과 같다]
# acc스코어 : 0.9736842105263158

# loss :  [0.9516311287879944, 0.359649121761322] ---------------------- 03_save_weights : 랜덤 가중치 저장
# acc스코어 : 0.35964912280701755

# loss :  [0.14472554624080658, 0.9736841917037964]--------------------- 04_save_weights : 훈련된 가장 좋게 저장된 가중치  [기존 훈련의 가장 좋은 값과 같다]
# acc스코어 : 0.9736842105263158








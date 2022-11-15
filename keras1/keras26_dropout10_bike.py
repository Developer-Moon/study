import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_breast_cancer  
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score, mean_squared_error  
from sklearn.metrics import r2_score 
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  


#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

#1. 데이타 
path = './_data/kaggle_bike/'      
                                  
train_set = pd.read_csv(path + 'train.csv', index_col=0)                       
print(train_set)
print(train_set.shape) #(10886, 11) 컬럼 11개



test_set = pd.read_csv(path + 'test.csv', index_col=0)                                   
print(test_set)        
print(test_set.shape) 



print(train_set.columns)     # (6493, 8)
print(train_set.info())      
print(train_set.describe()) 


print(train_set.isnull().sum()) 
test_set = test_set.fillna(test_set.mean())  # 결측지처리 nan 값에 0 기입   추가코드
train_set = train_set.dropna()  
print(train_set.isnull().sum())
print(train_set.shape)     


x = train_set.drop(['casual', 'registered', 'count'], axis=1)   #drop 뺀다         axis=1 열이라는걸 명시
print(x)
print(x.columns) #[10886 rows x 8 columns]
print(x.shape)   #(10886, 8)   input_dim=8

y = train_set['count']   #카운트 컬럼만 빼서 y출력---------- (sampleSubmission.csv 에서 구하려고 하는값이 count값이라서?? )
print(y)  
print(y.shape)  #(10886,) 10886개의 스칼라  output 개수 1개       여기까지 #1 데이터 부분을 잡은것

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.99,
    shuffle=True,     # 12 = 124  15까지 했음
    random_state=5
    )



scaler = RobustScaler()

scaler.fit(x_train)                      # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)        # train이 변한 범위에 맞춰서 변환됨
test_set = scaler.transform(test_set)  
# y_summit = model.predict(test_set) test셋은 스케일링이 상태가 아니니 summit전에 스케일링을 해서  y_summit = model.predict(test_set) 에 넣어줘야 한다 
# summit하기 전에만 해주면 상관이 없다

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위





#2. 모델구성
input_01 = Input(shape=(8,))
dense_01 = Dense(100)(input_01)
dense_02 = Dense(100, activation='relu')(dense_01)
dense_03 = Dense(100, activation='relu')(dense_02)
dense_04 = Dense(100)(dense_03)
dense_05 = Dense(100)(dense_04)
dense_06 = Dense(100)(dense_05)
dense_07 = Dense(100, activation='relu')(dense_06)
dense_08 = Dense(100, activation='relu')(dense_07)
dense_09 = Dense(100)(dense_08)
dense_10 = Dense(100)(dense_09)
dense_11 = Dense(100)(dense_10)
dense_12 = Dense(100, activation='relu')(dense_11)
output_01 = Dense(1)(dense_12)
model = Model(inputs=input_01, outputs=output_01)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])                                                 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=300, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)     

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))  





rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           




# y_summit = model.predict(test_set)
# y_summit = abs(y_summit)
# print(y_summit)
# print(y_summit.shape) # (715, 1)




# sampleSubmission = pd.read_csv('./_data/kaggle_bike/sampleSubmission.csv')
# sampleSubmission['count'] = y_summit
# print(sampleSubmission)
# sampleSubmission.to_csv('./_data/kaggle_bike/sampleSubmission_m.csv', index = False)






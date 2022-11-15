import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_breast_cancer  
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split   
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
path = './_data/ddarung/'                                        
train_set = pd.read_csv(path + 'train.csv',                      
                        index_col=0)                             

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)       


#### 결측치 처리 1. 제거 (이렇게 하는건 멍청한거다) ####

print(train_set.isnull().sum()) #널이 있는 곳에     널의 합계를 구한다?
test_set = test_set.fillna(test_set.mean())  # 결측지처리 nan 값에 0 기입   추가코드
train_set = train_set.dropna()  #행별로 싹 날려뿌겠다
print(train_set.isnull().sum())
print(train_set.shape)     #(1328, 10)   결측치 130개 정도 지워짐 


 
x = train_set.drop(['count'], axis=1)   #drop 뺀다         axis=1 열이라는걸 명시  axis=0  행 
print(x)
print(x.columns) #[1459 rows x 9 columns]
print(x.shape)   #(1459, 9)   input_dim=9

y = train_set['count']   #카운트 컬럼만 빼서 y출력
print(y)  
print(y.shape)  #(1459,) 1459개의 스칼라  output 개수 1개       여기까지 #1 데이터 부분을 잡은것

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9,
    shuffle=True, 
    random_state=3
    )



scaler = MinMaxScaler() 
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




#2. 모델구성             시퀀셜모델, 함수형 모델이 있다.

input_01 = Input(shape=(9,))
dense_01 = Dense(500)(input_01)
dense_02 = Dense(110, activation='relu')(dense_01)
dense_03 = Dense(120, activation='relu')(dense_02)
dense_04 = Dense(130, activation='relu')(dense_03)
dropout_01 = Dropout(0.2)(dense_04)
dense_05 = Dense(140, activation='relu')(dropout_01)
dense_06 = Dense(150, activation='relu')(dense_05)
dense_07 = Dense(140)(dense_06)
dense_08 = Dense(150, activation='relu')(dense_07)
dense_09 = Dense(140, activation='relu')(dense_08)
dropout_02 = Dropout(0.5)(dense_09)
dense_10 = Dense(150, activation='relu')(dropout_02)
dense_11 = Dense(140, activation='relu')(dense_10)
dense_12 = Dense(150)(dense_11)
dense_13 = Dense(140)(dense_12)
dense_14 = Dense(150)(dense_13)
dropout_03 = Dropout(0.2)(dense_14)
dense_15 = Dense(150, activation='relu')(dropout_03)
dense_16 = Dense(140, activation='relu')(dense_15)
dense_17 = Dense(150)(dense_16)
dense_18 = Dense(140)(dense_17)
dense_19 = Dense(150)(dense_18)
dense_20 = Dense(140)(dense_19)
dropout_04 = Dropout(0.5)(dense_20)
dense_21 = Dense(150)(dropout_04)
output_01 = Dense(1)(dense_21)
model = Model(inputs=input_01, outputs=output_01)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])                                                  
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)

from sklearn.metrics import r2_score, mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)  

# y_summit = model.predict(test_set)

# submission = pd.read_csv('./_data/ddarung/submission.csv')
# submission['count'] = y_summit
# print(submission)
# submission.to_csv('./_data/ddarung/submission2.csv', index = False)


# r2스코어 : 0.6035844061804955
# RMSE : 56.65080837027634
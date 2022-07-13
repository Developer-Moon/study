import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_breast_cancer  
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten
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

print(x.shape)
print(y.shape)



x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.9,
    shuffle=True, 
    random_state=3
    )

from sklearn.preprocessing import MaxAbsScaler,RobustScaler, MinMaxScaler, StandardScaler

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(1195, 3, 3, 1)
x_test = x_test.reshape(133, 3, 3, 1)


#2. 모델구성         
input_01 = Input(shape=(3, 3, 1))
conv2D_01 = Conv2D(64, (1,1))(input_01)
conv2D_02 = Conv2D(32, (1,1), activation='relu')(conv2D_01)
dropout_01 = Dropout(0.2)(conv2D_02)
conv2D_03 = Conv2D(4, (1,1), activation='relu')(dropout_01)
dropout_02 = Dropout(0.2)(conv2D_03)
flatten_01 = Flatten()(dropout_02)
dense_01 = Dense(32, activation='relu')(flatten_01)
dense_02 = Dense(32, activation='relu')(dense_01)
output_01 = Dense(1)(dense_02)
model = Model(inputs=input_01, outputs=output_01)



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


# r2스코어 : 0.7530413363160167
# RMSE : 44.7139338080419


import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_breast_cancer  
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score, mean_squared_error  
from sklearn.metrics import r2_score 
import numpy as np 
import time
import pandas as pd

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

#2. 모델구성
#2. 모델구성
model = Sequential()
model.add(Dense(100,input_dim=8))      
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse'])                                                 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=300, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)     

y_predict = model.predict(x_test) 

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))  




rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           




y_summit = model.predict(test_set)
y_summit = abs(y_summit)
print(y_summit)
print(y_summit.shape) # (715, 1)




sampleSubmission = pd.read_csv('./_data/kaggle_bike/sampleSubmission.csv')
sampleSubmission['count'] = y_summit
print(sampleSubmission)
sampleSubmission.to_csv('./_data/kaggle_bike/sampleSubmission_m.csv', index = False)











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


# loss :  0.11895273625850677
# val_loss: 0.1199
# r2스코어 : 0.4834914503626927


# sigmoid 쓸때
# loss :  0.15451954305171967
# val_loss: 0.0693
# r2스코어 : 0.7946649335930135




#과제 보스턴, 켈리포니아, 디아벳 액티베이션 활용해서 성능 향상시키기
# 따릉이 바이크 부동산
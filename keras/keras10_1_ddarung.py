# 데이콘 따릉이 문제풀이


import numpy as np                                               
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error          #, mean_squared_error 이게 rmse

#1. 데이타 
path = './_data/ddarung/'                                         # path(변수)에 경로를 넣음
train_set = pd.read_csv(path + 'train.csv',                       # 판다스로 csv(엑셀시트)파일을 읽어라   path(경로) + train.csv
                        index_col=0)                              # id가 첫번째로 와라

print(train_set)
print(train_set.shape) #(1459, 10) 컬럼 10개(인덱스 제외)

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)                               #이 값은 예측 부분에서 쓴다

print(test_set)        #[715 rows x 9 columns]
print(test_set.shape)  #(715, 9)



print(train_set.columns)     # 컬럼
print(train_set.info())  
print(train_set.describe())  # 판다스로 땡겨왔기 때문에 DESCR보다 더 많은 정보들을 준다


#### 결측치 처리 1. 제거 (이렇게 하는건 멍청한거다) ####

print(train_set.isnull().sum()) #널이 있는 곳에     널의 합계를 구한다?
test_set = test_set.fillna(test_set.mean())  # 결측지처리 nan 값에 0 기입   추가코드
train_set = train_set.dropna()  #행별로 싹 날려뿌겠다
print(train_set.isnull().sum())
print(train_set.shape)     #(1328, 10)   결측치 130개 정도 지워짐 


#                       컬럼당 널의 개수
# hour                        0 
# hour_bef_temperature        2
# hour_bef_precipitation      2
# hour_bef_windspeed          9
# hour_bef_humidity           2
# hour_bef_visibility         2
# hour_bef_ozone             76
# hour_bef_pm10              90
# hour_bef_pm2.5            117
# count                       0
# dtype: int64


###########################

 
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


#2. 모델구성
model = Sequential()
model.add(Dense(500,input_dim=9, activation='relu'))  #처음 output값이 1 이면 성능이 쓰레기다    
model.add(Dense(110))
model.add(Dense(120))
model.add(Dense(130))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))

model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')               #loss는 rmse를 제공하지 않는다고 한다
model.fit(x_train, y_train, epochs=500, batch_size=500)   #, verbose=0

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)     # 데이터셋에 값이 없는(nan?) 애들이 있어서 그 애들을 빼서 훈련시켜야한다 - 그런데 위험하고 무식한 작업이라고 한다(데이터가 많아서)

y_predict = model.predict(x_test) #예측해서 나온 값

def RMSE(y_test, y_predict):  # ()괄호안 함수를 받아들이겠다    나는 함수를 만들거다   이 함수는 y_test, y_predict이걸 받아 들일꺼다
    return np.sqrt(mean_squared_error(y_test, y_predict))  # 내가 받아들인   y_test, y_predict이걸     mean_squared_error여기다 넣는다   그리고 루트를 씌우겠다  그리고 리턴

#엠에스이가 제곱을 해서 다시 숫자가 커져서 (sqrt)루트를 씌우겠다 


rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           #로스에 루트 씌운값이 rmse다


# loss : 23.453350067138672
# RMSE : 29.564602970449258

y_summit = model.predict(test_set)

print(y_summit)
print(y_summit.shape) # (715, 1)

############# .to_csv() 함수를 사용해서 
############# submission 을 완성하시오!!!

submission = pd.read_csv('./_data/ddarung/submission.csv')
submission['count'] = y_summit
print(submission)
submission.to_csv('./_data/ddarung/submission2.csv', index = False)













# 과제1 함수에 대해서 공부해라 (다시 사용하기 위해 사용)





# y_predict = model.predict(test_set)











# dtypes: float64(9), int64(1)
# memory usage: 125.4 KB
# None
#               hour  hour_bef_temperature  hour_bef_precipitation  ...  hour_bef_pm10  hour_bef_pm2.5        count
# count  1459.000000           1457.000000             1457.000000  ...    1369.000000     1342.000000  1459.000000
# mean     11.493489             16.717433                0.031572  ...      57.168736       30.327124   108.563400
# std       6.922790              5.239150                0.174917  ...      31.771019       14.713252    82.631733
# min       0.000000              3.100000                0.000000  ...       9.000000        8.000000     1.000000
# 25%       5.500000             12.800000                0.000000  ...      36.000000       20.000000    37.000000
# 50%      11.000000             16.600000                0.000000  ...      51.000000       26.000000    96.000000
# 75%      17.500000             20.100000                0.000000  ...      69.000000       37.000000   150.000000
# max      23.000000             30.000000                1.000000  ...     269.000000       90.000000   431.000000

#mean : 평균    



# Int64Index: 1459 entries, 3 to 2179
# Data columns (total 10 columns):
    
#   Column                  Non-Null Count  Dtype            null 데이터가 없다(빠져있다)?     결측지(작업할때 이게 얼마나 있느냐를 판단해야한다)   y값은 결측이 없다
# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64











# 715개의 카운트 값을 뽑아내서 제출해라                             


# 트레인셋에서 모든걸 다 하고 테스트 셋에서 model.predict를 뽑아라

# 테스트 셋은 카운터가 빠져있다  제출용?




#        인덱스[id] - (연산X)     
#         id  hour  hour_bef_temperature  hour_bef_precipitation  ...  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5  count
# 0        3    20                  16.3                     1.0  ...           0.027           76.0            33.0   49.0
# 1        6    13                  20.1                     0.0  ...           0.042           73.0            40.0  159.0
# 2        7     6                  13.9                     0.0  ...           0.033           32.0            19.0   26.0
# 3        8    23                   8.1                     0.0  ...           0.040           75.0            64.0   57.0
# 4        9    18                  29.5                     0.0  ...           0.057           27.0            11.0  431.0
# ...    ...   ...                   ...                     ...  ...             ...            ...             ...    ...
# 1454  2174     4                  16.8                     0.0  ...           0.031           37.0            27.0   21.0
# 1455  2175     3                  10.8                     0.0  ...           0.039           34.0            19.0   20.0
# 1456  2176     5                  18.3                     0.0  ...           0.009           30.0            21.0   22.0
# 1457  2178    21                  20.7                     0.0  ...           0.082           71.0            36.0  216.0
# 1458  2179    17                  21.1                     0.0  ...           0.046           38.0            17.0  170.0
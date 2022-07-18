from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error   
import numpy as np                                               
import pandas as pd



#1. 데이터
amore = pd.read_csv('./_data/class_amore/amore.csv', encoding = 'CP949')
samsung = pd.read_csv('./_data/class_amore/samsung.csv', encoding = 'CP949')
print(amore.columns)
# ['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']
print(samsung.columns)
# ['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비']



path = './_data/ddarung/'                                         # path(변수)에 경로를 넣음

train_set = pd.read_csv(path + 'train.csv', index_col=0)          # 판다스로 csv(엑셀시트)파일을 읽어라   path(경로) + train.csv
                                                                  # index_col=0 - id가 첫번째로 와라
print(train_set)
print(train_set.columns)
# ['hour', 'hour_bef_temperature', 'hour_bef_precipitation', 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility', 'hour_bef_ozone', 'hour_bef_pm10',
# 'hour_bef_pm2.5', 'count']
print(train_set.shape) # (1459, 10) 컬럼 10개(인덱스 제외)


test_set = pd.read_csv(path + 'test.csv', index_col=0)            # 이 값은 예측 부분에서 쓴다
print(test_set.columns) 
# ['hour', 'hour_bef_temperature', 'hour_bef_precipitation', 'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility', 'hour_bef_ozone', 'hour_bef_pm10',
# 'hour_bef_pm2.5']       
print(test_set.shape) # (715, 9)



print(train_set.info())  
"""
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   hour                    1459 non-null   int64
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   float64
dtypes: float64(9), int64(1)
"""

print(train_set.describe())  # 평균치, 중간값, 최소값 등 출력
"""
             hour  hour_bef_temperature  hour_bef_precipitation  hour_bef_windspeed  ...  hour_bef_ozone  hour_bef_pm10  hour_bef_pm2.5        count
count  1459.000000           1457.000000             1457.000000         1450.000000  ...     1383.000000    1369.000000     1342.000000  1459.000000
mean     11.493489             16.717433                0.031572            2.479034  ...        0.039149      57.168736       30.327124   108.563400
std       6.922790              5.239150                0.174917            1.378265  ...        0.019509      31.771019       14.713252    82.631733
min       0.000000              3.100000                0.000000            0.000000  ...        0.003000       9.000000        8.000000     1.000000
25%       5.500000             12.800000                0.000000            1.400000  ...        0.025500      36.000000       20.000000    37.000000
50%      11.000000             16.600000                0.000000            2.300000  ...        0.039000      51.000000       26.000000    96.000000
75%      17.500000             20.100000                0.000000            3.400000  ...        0.052000      69.000000       37.000000   150.000000
max      23.000000             30.000000                1.000000            8.000000  ...        0.125000     269.000000       90.000000   431.000000
"""



#### 결측치 처리 1. 제거 (이렇게 하는건 멍청한거다) ####
# 데이터셋에 값이 없는(NaN) 애들이 있어서 그 애들을 빼서 훈련시켜야한다 - 그런데 위험하고 무식한 작업이라고 한다(데이터가 많아서)

print(train_set.isnull().sum()) # null 의 개수 확인, isnull()만 쓸경우 null값이라면 True 아니라면 False를 출력
"""
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
"""

print(test_set.isnull().sum())
"""
hour                       0
hour_bef_temperature       1
hour_bef_precipitation     1
hour_bef_windspeed         1
hour_bef_humidity          1
hour_bef_visibility        1
hour_bef_ozone            35
hour_bef_pm10             37
hour_bef_pm2.5            36
dtype: int64
"""
print(train_set.shape)   # (1459, 10)            
print(test_set.shape)    # (715, 9)
train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   # fillna() - 결측값을 (특정값)로 채우겠다
                                              # 결측값을 결측값의 앞 행의 값으로 채우기 : df.fillna(method='ffill') or df.fillna(method='pad')
                                              # 결측값을 결측값의 뒷 행의 값으로 채우기 : df.fillna(method='bfill') or df.fillna(method='backfill')
                                              # 결측값을 각 열의 평균 값으로 채우기     : df.fillna(df.mean())
                                              
print(train_set.shape)  # (1459, 10)            
print(test_set.shape)   # (715, 9)
print(train_set.isnull().sum())               # train 결측지 평균값으로 채움                                     
print(test_set.isnull().sum())                # test 결측지 평균값으로 채움      
    
                                             
train_set = train_set.dropna()                # dropna() - 행별로 싹 날려뿌겠다 : 결측지를 제거 하는 법[위 에서 결측지를 채워서 지금은 의미 없다]
                                              # 결측값 있는 행 제거 : df.dropna() or df.dropna(axis=0)
                                              # 결측값 있는 열 제거 : df.dropna(axis=1)

print(train_set.shape)  # (1459, 10)         
print(test_set.shape)   # (715, 9)                                 



x = train_set.drop(['count'], axis=1)         # train_set에서 count를 drop(뺀다) axis=1 열, axis=0 행 
print(x)
print(x.columns)                              # [1459 rows x 9 columns]
print(x.shape)                                # (1459, 9) - input_dim=9

y = train_set['count']                        # y는 train_set에서 count컬럼이다
print(y)  
print(y.shape)                                # (1459,) 1459개의 스칼라  output=1    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=16)



#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=9))   
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(140))
model.add(Dense(150))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                # loss에선 rmse를 제공하지 않는다
model.fit(x_train, y_train, epochs=300, batch_size=100)   



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)     

y_predict = model.predict(x_test)                          # 예측해서 나온 값

def RMSE(y_test, y_predict):                               # 이 함수는 y_test, y_predict를 받아 들인다
    return np.sqrt(mean_squared_error(y_test, y_predict))  # 내가 받아들인 y_test, y_predict를 mean_squared_error에 넣는다 그리고 루트를 씌운다 그리고 리턴
                                                           # mse가 제곱하여 숫자가 커져서 (sqrt)루트를 씌우겠다 
rmse = RMSE(y_test, y_predict)  
print("RMSE :", rmse)           



# y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (715, 1)

# submission = pd.read_csv('./_data/ddarung/submission.csv')
# submission['count'] = y_summit
# print(submission)
# submission.to_csv('./_data/ddarung/submission2.csv', index = False)


# loss : 3058.065673828125
# RMSE : 55.299774499917866
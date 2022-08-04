from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error          # mean_squared_error : RMSE
import numpy as np                                               
import pandas as pd

from sklearn.svm import LinearSVR

#1. 데이타 
path = './_data/ddarung/'                                         # path(변수)에 경로를 넣음
train_set = pd.read_csv(path + 'train.csv', index_col=0)          # 판다스로 csv(엑셀시트)파일을 읽어라   path(경로) + train.csv                                                               
test_set = pd.read_csv(path + 'test.csv', index_col=0)            # 이 값은 예측 부분에서 쓴다   

print(test_set.shape) # (715, 9)

train_set = train_set.fillna(train_set.mean())
test_set = test_set.fillna(test_set.mean())   # fillna() - 결측값을 (특정값)로 채우겠다
                                              # 결측값을 결측값의 앞 행의 값으로 채우기 : df.fillna(method='ffill') or df.fillna(method='pad')
                                              # 결측값을 결측값의 뒷 행의 값으로 채우기 : df.fillna(method='bfill') or df.fillna(method='backfill')
                                              # 결측값을 각 열의 평균 값으로 채우기     : df.fillna(df.mean())
                                              
print(train_set.isnull().sum())               # train 결측지 평균값으로 채움                                     
print(test_set.isnull().sum())                # test 결측지 평균값으로 채움      
    
                                             
train_set = train_set.dropna()                # dropna() - 행별로 싹 날려뿌겠다 : 결측지를 제거 하는 법[위 에서 결측지를 채워서 지금은 의미 없다]
                                              # 결측값 있는 행 제거 : df.dropna() or df.dropna(axis=0)
                                              # 결측값 있는 열 제거 : df.dropna(axis=1)

x = train_set.drop(['count'], axis=1)         # train_set에서 count를 drop(뺀다) axis=1 열, axis=0 행 
print(x)
print(x.columns)                              # [1459 rows x 9 columns]
print(x.shape)                                # (1459, 9) - input_dim=9

y = train_set['count']                        # y는 train_set에서 count컬럼이다
print(y)  
print(y.shape)                                # (1459,) 1459개의 스칼라  output=1    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=16)



#2. 모델구성
model = LinearSVR()


#3. 컴파일, 훈련
model.fit(x_train, y_train) 



#4. 평가, 예측




results = model.score(x_test, y_test)
print('r2 :', results)     
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

# 머신러닝 사용 
# r2 : 0.4933842411244941
# RMSE : 58.89354387868726
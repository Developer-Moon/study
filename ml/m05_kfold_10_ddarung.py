from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict  # Kfold - cross_val_score검증하기위해 이걸 쓴다
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVR, LinearSVC
import numpy as np
import pandas as pd


# 1. 데이터
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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=9)
                      
n_splits =5                # n_splits=5 5등분
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66 ) #         
                    
                    
                      
#2. 모델구성
model = SVR()



#3. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model, x_train, y_train, cv=kfold) # cv=5 라면 kfold를 5로 쓴다
print('ACC :', scores, '\ncross_val_score :' , round(np.mean(scores), 4)) # 4번째까지 출력 (반올림을 5번째 자리에서)

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)

r2 = r2_score(y_test, y_predict)
print('cross_val_predict r2 : ', r2)

# cross_val_predict r2 :  0.037694879079877786
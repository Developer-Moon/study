# [실습] 아래를 완성할 것
# 1. train 0.7
# 2. R2 0.8 이상

from tensorflow.python.keras.models import Sequential   
from tensorflow.python.keras.layers import Dense 
import numpy as np 
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   
from sklearn.metrics import r2_score 



#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      

print(x)                  
print(y)                  

print(x.shape, y.shape)          #(506, 13) (506,) 506개의 데이터 개수   13개의 컬럼 (input_dim=13)      (506개의 스칼라 1개의 벡터)

print(datasets.feature_names)    # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] 컬럼셋 b는 흑인 그래서 이 데이터 셋은 못 쓰게한다

print(datasets.DESCR)            #DESCR 설명하다 묘사하다 - 컬럼들의 소개가 나온다


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7,
    shuffle=True,
    random_state=66
    )



#2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=13))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))



import time

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                   

start_time = time.time()                                        #현재시간을 보여준다
print(start_time)                                               # 스타트지점 1656033369.6949103
model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=0) # verbose = 훈련과정을 보여주지 않는다 -> 바로 결과가 나온다                                
                                                                # 사람에게 훈련과정을 보여주게 위해 딜레이를 걸어 시간이 걸린다 그 과정을 안하기 위해 쓴다
end_time = time.time() - start_time 

print("걸린시간 : ", end_time)                                   # 10.109211683273315   

"""
verbose 0일때 걸린시간 : 10.109211683273315 / 출력이 없다.

verbose 1일때 걸린시간 : 12.079090356826782 / 잔소리많다훈련과정을 보여주기 위해 2초 이상의 시간이 걸렸다.

verbose 2일때 걸린시간 : 10.555930614471436 / 프로그래스바가 없어졌다

verbose 3일때 걸린시간 : 10.417572975158691 / 이제는 epochs만 나온다

verbose 4일때 걸린시간 : 10.422906160354614 / 3일때와 동일 

verbose는 시간을 줄이는데 좋다 하지만  loss 및 정보들을 볼 수가 없다 작업상황에 따라 조절하여 쓰든가...
"""



# #4. 평가 예측
# loss = model.evaluate(x_test, y_test)
# print('loss :', loss)

# y_predict = model.predict(x_test)

# r2 = r2_score(y_test, y_predict)
# print('r2스코어 :', r2)



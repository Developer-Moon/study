import numpy as np   
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense    

from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라
x_train, x_test, y_train, y_test = train_test_split(            
    x, y, train_size=0.7,
    test_size=0.3,
    # shuffle=Ture, 디폴트값
    random_state=66
)
# 테스트 사이즈 0.3 트레인 사이즈 0.7 셔플 = 섞을것이다   random_state : 랜덤난수(난수값- 난수표의 66번 값을 써라 - 바뀔때마다 이렇게 바뀐다?)
print(x_train) #[2 7 6 3 4 8 5]
print(y_test)  #[ 1  9 10]
print(x_train) #[2 7 6 3 4 8 5]
print(y_test)  #[ 1  9 10]





#                                                                                                             #함수 : 재사용을 염두를 두고 기능을 모아둔 것
#                                                                                                             #사이킷 런
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=777, shuffle=False)   #shuffle=False 순차적으로 반환, 기입하지 않으면 랜덤 난수(검색) 반환
#                                                                          # 당최 random_state 이건 먼가??? 
# print(x_train)
# print(y_test)
# print(x_train)
# print(y_test)








#과제 넘파이 리스트의 슬라이싱!! 7:3으로 잘라라

# x_train = x[0:7]    # = x[:7]   이런식으로 절대 안 자른다        30%를 자른다면 [1, 2, 4, 5, 7, 8, 9] 중간에 랜덤하게 뺀다 (셔플) 하는 이유 -  
# x_test = x[0:7]     # = x[7:]                                  이런식으로 작업하면 일부를 변화 시킬때 나눈지점에서 기울기가 틀어진다 그러지 않기 위해 셔플하여 자른다?
# y_train = x[7:10]
# y_test = x[7:10]




#2. 모델구성
model = Sequential()  
model.add(Dense(10, input_dim=1))      
model.add(Dense(1))
 
 
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)
 
 
 #4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss :', loss)
result = model.predict([11]) 
print('11의 예측값은 :', result)

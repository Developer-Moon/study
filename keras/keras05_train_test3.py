import numpy as np   
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense    

from sklearn.model_selection import train_test_split 

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) 

#[검색] train과 test를 섞어서 7:3으로 찾을 수 있는 방법을 찾아라
x_train, x_test, y_train, y_test = train_test_split(                 # 데이터셋 전체를 훈련의 범위로 잡고 그 중 일부를 빼서 섞어(셔플) 70프로를 훈련시킨다
    x, y, train_size=0.7,                                            # 트레인과 테스트 셋은 간섭이 없어야 한다
    test_size=0.3,
    shuffle=True,                                                    # shuffle=True, 디폴트값 or shuffle=False(셔플하지 않는다)
    random_state=1004                                                # random_state : 랜덤난수(난수값- 난수표의 66번 값을 써라), 기입하지 않으면 랜덤 난수 반환
)

print(x_train) #[2 7 6 3 4 8 5]
print(y_test)  #[ 1  9 10]
print(x_train) #[2 7 6 3 4 8 5]
print(y_test)  #[ 1  9 10]


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

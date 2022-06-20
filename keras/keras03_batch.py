import numpy as np    
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense   

#1. 데이터
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])


#2. 모델구성
model = Sequential()
model.add(Dense(2, input_dim=1))                       
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')            
model.fit(x, y, epochs=1000, batch_size=1)              # batch_size=1 작업하는데 1개씩 잘라서 작업하겠다 (하이퍼 파라미터) - 쓰는이유? : 메모리 및 그래픽카드 때문?
                                                        # 단점 : size 값이 낮을수록 시간이 오래 걸린다
                                                        # 장점 : 메모리가 할당량이 작게 잡힌다, 훈련량이 많아져 loss값이 작아지고 weight값이 정확해진다
                                                          



#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])  
print('예측값은 : ', result)

# loss :  0.4005308151245117
# 예측값은 :  [[5.990535]]
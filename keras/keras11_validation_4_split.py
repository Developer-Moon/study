from sklearn.model_selection import train_test_split                                # 발리데이션을 스플릿으로 나눌 수 있다
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
import numpy as np  


#1. 데이터 
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x,y,
    test_size=0.2, random_state=66
)

print(x_train.shape, x_test.shape) #(12,) (4,)




#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25) # 훈련하고 문제풀고 훈련하고 문제풀고     validation_split=0.25 트레인셋에서 25프로가 validation 개수


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)                                         # 여기서 수능보고 
print('loss : ', loss)

result = model.predict([17])
print('17의 예측값 : ', result)

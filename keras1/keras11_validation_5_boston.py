from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score 
from sklearn.datasets import load_boston  


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      
                  
print(x.shape, y.shape)   # (506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)



#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=13))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')                   
model.fit(x_train, y_train, epochs=300, batch_size=10, validation_split=0.2)



#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print('loss :', loss)
print('r2 :', r2)

"""
loss : 19.609092712402344 ----- Normal
r2 : 0.7626509148591476

loss : 24.405136108398438 ----- validation_split
r2 : 0.704599447055035

"""


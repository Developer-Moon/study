import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_diabetes  
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D 
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler 

   
#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

#1. 데이터
datasets = load_diabetes()   
x = datasets.data
y = datasets.target      

print(x.shape) # (442, 10)
print(y.shape) # (442,)




x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )

print(x_train.shape) # (353, 10)
print(x_test.shape)  # (89, 10)

x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)





#2. 모델구성
model = Sequential()
model.add(Conv1D(200,2,activation='relu', input_shape=(10,1))) 
model.add(Flatten())
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=200, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)



# loss : 3711.32373046875          기존훈련
# r2스코어 : 0.42815090581560145

# loss : 3313.83349609375          dropout
# r2스코어 : 0.48939712579303074


# Conv1D
# loss : 3247.035888671875
# r2스코어 : 0.49968940423144903
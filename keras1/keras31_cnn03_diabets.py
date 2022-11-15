import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_diabetes  
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
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

x = x.reshape(442, 10, 1, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )





#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(1, 1), input_shape=(10, 1, 1)))
model.add(MaxPooling2D(1, 1))    #(14, 14, 64) 
model.add(Conv2D(32, (1, 1),
                 padding='valid',          # 디폴트
                 activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(1, 1))    #(14, 14, 64) 
model.add(Conv2D(4, (1, 1),
                 padding='valid',          # 디폴트
                 activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten()) # (N, 252)                                                                         
model.add(Dense(32, activation='relu'))                                                                                                         
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()







#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)



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


# loss : 3524.6474609375
# r2스코어 : 0.45691440913714665


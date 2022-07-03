import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_diabetes  
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score 
import numpy as np 
import time

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


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )

#2. 모델구성
#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(100))  
model.add(Dense(100))  
model.add(Dense(100))  
model.add(Dense(100))  
model.add(Dense(100))  
model.add(Dense(10))  
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])                                                 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=500, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score 
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)


# 메트릭스를 쓰면 로스가 두개가 나온다   앞에껀 binary_crossentropy 뒤에껀 에큐러시(정확도)
# loss :  [0.2551944851875305, 0.9122806787490845]   

# 이진분류에서는 mse를 신뢰할 수 없다








import matplotlib.pyplot as plt    
plt.figure(figsize=(9,6))                                                   
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')           
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss') 


plt.grid()            
plt.title('안결바보') 
plt.xlabel('epochs')
plt.ylabel('loss')

plt.legend() 
plt.show()


# loss :  3197.76123046875
# val_loss: 3192.9006
# r2스코어 : 0.5072817321713152

####################################
# 20220703

# loss : 3807.1777
# val_loss : 3079.1750 
# mae : 48.9231
# val_mae : 45.8657
# r2스코어 : 0.41338157521308094
####################################


#과제 보스턴, 켈리포니아, 디아벳 액티베이션 활용해서 성능 향상시키기
# 따릉이 바이크 부동산
import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_boston   
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
datasets = load_boston()
x = datasets.data
y = datasets.target      


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )



#2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=13))
model.add(Dense(20, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
# model.add(Dense(1, activation='sigmoid')) 


#3. 컴파일, 훈련
model.compile(loss='mse',        # 분류 모델중 이진 분류는 무조건 binary_crossentropy를 쓴다 - 보스턴은 회귀모델 이므로 
              optimizer='adam',
              metrics=['mae'])     # metrics=평가지표를 판단   받아들이는게 리스트 형태 ['accuracy', 'mse']        
                                    # True 또는 False, 양성 또는 음성 등 2개의 클래스를 분류할 수 있는 분류기를 의미                                                                                                                                                                                                                                                                           
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=1000, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score # 두개 같이 쓸 수 있다 
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


# loss :  0.11895273625850677
# val_loss: 0.1199
# r2스코어 : 0.4834914503626927




####################################
# 20220703

# loss: loss : 13.3111
# val_loss : 35.0299
# mae : 2.6359
# val_mae : 3.7203
# r2스코어 : 0.8407432260663265
####################################


#과제 보스턴, 켈리포니아, 디아벳 액티베이션 활용해서 성능 향상시키기
# 따릉이 바이크 부동산
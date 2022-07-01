import numpy as np
from sklearn import metrics         
from sklearn.datasets import load_breast_cancer  
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
datasets = load_breast_cancer()
print(datasets.DESCR)
# Number of Instances: 569 (행)
# Number of Attributes: 30  (열)   (569,30)
# print(datasets.feature_names)

x = datasets.data   # x = datasets['data'] 으로도 쓸 수 있다.
y = datasets.target 
print(x.shape, y.shape) # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))   #  activation=sigmoid        linear 선형 - 
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='relu'))    # activation='relu 중간에서만 쓸 수 있다 히든에서만
model.add(Dense(100, activation='linear'))  # sigmoid 중간에 한 두개씩 넣어보기
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1, activation='sigmoid')) # sigmoid를 쓰면 무조건 0과 1 사이의 값이 나온다  그 이후 loss='binary_crossentropy'사용


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])  # 분류 모델중 이진 분류는 무조건 binary_crossentropy를 쓴다           
                                                             # True 또는 False, 양성 또는 음성 등 2개의 클래스를 분류할 수 있는 분류기를 의미
                                                             
                                                             # metrics=평가지표를 판단   받아들이는게 리스트형태 ['accuracy', 'mse']
                                                             # 메트릭스를 넣으면 로스 이외에 다른 지표도 나온다
                                                             # 회귀모델일때 메트릭스에  로스는 mas쓰고  etrics=['mae'] 이걸로 두개 확인 가능                                                 
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 이것도 리스트 형태 2가지 이상



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, accuracy_score # 두개 같이 쓸 수 있다 
# r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)  # print(y_predict) y_test 여기에 들어가는 값이 딱 떨어져야 한다
print('r2스코어 :', acc)


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


# sigmoid 쓸때
# loss :  0.15451954305171967
# val_loss: 0.0693
# r2스코어 : 0.7946649335930135




#과제 보스턴, 켈리포니아, 디아벳 액티베이션 활용해서 성능 향상시키기
# 따릉이 바이크 부동산
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input 
from sklearn.model_selection import train_test_split                                     
from sklearn.datasets import load_breast_cancer  
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np 
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler   

#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

#1. 데이터 
datasets = load_breast_cancer()
x = datasets.data              # x = datasets['data'] 으로도 쓸 수 있다.
y = datasets.target 
print(x.shape, y.shape)        # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.8,
    random_state=66
    )




               


# scaler = MinMaxScaler() 
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train)     # 스케일링을 했다.
x_train = scaler.transform(x_train)      # 변환해준다  x_train이 0과 1사이가 된다.
x_test = scaler.transform(x_test)       # train이 변한 범위에 맞춰서 변환됨

print(np.min(x_train))  # 0.0                   0과 1사이
print(np.max(x_train))  # 1.0000000000000002    0과 1사이
print(np.min(x_test))   # -0.06141956477526944  0미만
print(np.max(x_test))   # 1.1478180091225068    0초과 범위



#2. 모델구성
input_01 = Input(shape=(30,))
dense_01 = Dense(100)(input_01)
dense_02 = Dense(100, activation='relu')(dense_01)
dense_03 = Dense(100)(dense_02)
dense_04 = Dense(100)(dense_03)
dense_05 = Dense(100)(dense_04)
dense_06 = Dense(100)(dense_05)
dense_07 = Dense(100, activation='sigmoid')(dense_06)
dense_08 = Dense(100)(dense_07)
output_01 = Dense(1, activation='sigmoid')(dense_08)
model = Model(inputs=input_01, outputs=output_01)
model.summary()


""" 기존 Sequential모델
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=30))  
model.add(Dense(100, activation='relu'))                 
model.add(Dense(100, activation='linear'))
model.add(Dense(100, activation='linear'))                   
model.add(Dense(100, activation='linear'))  
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='linear'))
model.add(Dense(1, activation='sigmoid'))    
model.summary()             
"""                 
 
                                        
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',   # 분류 모델중 이진 분류는 무조건 binary_crossentropy를 쓴다    
              optimizer='adam',
              metrics=['accuracy'])  # metrics = 평가지표를 판단, 리스트형태(2개 이상) ['accuracy', 'mse'] 에큐러시는 분류모델일때만 쓴다   
                                            # True 또는 False, 양성 또는 음성 등 2개의 클래스를 분류할 수 있는 분류기를 의미?? 어디선가 놓친부분??
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 리스트 형태



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print('loss : ', loss) 
print(hist.history['val_loss'])

y_predict = model.predict(x_test)

y_predict = y_predict.flatten()            
y_predict = np.where(y_predict > 0.5, 1 , 0) #0.5보다크면 1, 작으면 0

# y_predict[(y_predict<0.5)] = 0  
# y_predict[(y_predict>=0.5)] = 1 

print(y_predict)

from sklearn.metrics import r2_score, accuracy_score # 두개 같이 쓸 수 있다 
# r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)


"""""""""""""""""""""""""""""""""
[scaler = MinMaxScaler]

loss     : 0.44406625628471375
val_loss :
mae      : 
val_mae  : 
acc스코어 : 0.9385964912280702
"""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""
[scaler = StandardScaler]             함수모델 사용 

loss     : 0.3714667856693268
mae      : 0.6448
val_mae  : 3.0111                    loss : 0.3256046772003174
acc스코어 : 0.956140350877193        acc스코어 : 0.9649122807017544



[scaler = RobustScaler]

loss      : 0.4551304578781128
acc스코어 : 0.956140350877193



[scaler = RobustScaler]

loss      : 0.22564372420310974
acc스코어 : 0.9385964912280702
"""""""""""""""""""""""""""""""""












from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense 
from sklearn.model_selection import train_test_split                                     
from sklearn.datasets import load_breast_cancer  
from sklearn.metrics import r2_score
from sklearn import metrics
import numpy as np 
import time
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus) :
    print('쥐피유 돌아유')
    aaa = 'gpu - 쥐피유 돌아유'
else:
    print('내가 돌아유')
    aaa = 'cpu - 내가 돌아유'




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

#2. 모델구성
model = Sequential()
model.add(Dense(500, activation='linear', input_dim=30))   # activation=sigmoid       디폴트값 = linear 선형
model.add(Dense(400, activation='relu'))                   # activation='relu 는 히든에서만 사용가능
model.add(Dense(400, activation='relu'))                   # activation='relu 는 히든에서만 사용가능
model.add(Dense(400, activation='relu'))                   # activation='relu 는 히든에서만 사용가능
model.add(Dense(300, activation='linear'))
model.add(Dense(400, activation='linear'))                   
model.add(Dense(1, activation='sigmoid'))                  # sigmoid를 쓰면 무조건 0과 1 사이의 값이 나온다 (분류모델중 이중분류 일때만 사용가능) 
                                                           # 그 이후 compile쪽에서 loss='binary_crossentropy'사용
start_time = time.time()
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',   # 분류 모델중 이진 분류는 무조건 binary_crossentropy를 쓴다    
              optimizer='adam',
              metrics=['accuracy', 'mse'])  # metrics = 평가지표를 판단, 리스트형태(2개 이상) ['accuracy', 'mse'] 에큐러시는 분류모델일때만 쓴다   
                                            # True 또는 False, 양성 또는 음성 등 2개의 클래스를 분류할 수 있는 분류기를 의미?? 어디선가 놓친부분??
                                         
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[earlyStopping], verbose=1)   # callbacks=[earlyStopping] 리스트 형태



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

end_time = time.time() - start_time
print('loss : ', loss) 
print(hist.history['val_loss'])
print(aaa, '걸린시간 : ', end_time)

# cpu 77.2216739654541

# gpu 191.17069268226624








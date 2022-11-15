from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dense  
from sklearn.model_selection import train_test_split  
from sklearn.datasets import load_boston   


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target      

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)



#2. 모델구성
model = Sequential()
model.add(Dense(14, input_dim=13))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')       
from tensorflow.python.keras.callbacks import EarlyStopping      #얼리 스타핑을 적용하기 위해          #  EarlyStopping  (대문자니까) 클래스 
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)          
                # monitor='val_loss',(발로스를 중심으로, 로스 중심으로 힐려면 loss로)   partition=10, 10번 참을꺼야      
                # mode='min' 최소값   디폴트값은 auto  max는 에큐러시 할때 쓴다??                                                    
              
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, verbose=1, validation_split=0.2, callbacks=[earlyStopping])  
 

    


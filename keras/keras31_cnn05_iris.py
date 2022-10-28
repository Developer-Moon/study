from re import X
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten 
from sklearn.model_selection import train_test_split        
from sklearn.datasets import load_iris                              
from sklearn.metrics import r2_score
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler  

#plt 폰트 깨짐 현상 #
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#plt 폰트 깨짐 현상 #

import tensorflow as tf            
tf.random.set_seed(66)         #텐서플로의 난수표 66번 사용 이 난수는 w = 웨이트


#1. 데이터
datasets = load_iris()
print(datasets.DESCR)


print(datasets.feature_names)
x = datasets['data']
y = datasets['target']
print(x)
print(y)

print(x.shape, y.shape)        # (150, 4) (150,)   
print('y의 라벨값 :', np.unique(y))   # y의 라벨값 : [0 1 2] - 판다스에서 있다 : y값을 (150, 3) 으로 바꿔준다

from tensorflow.keras.utils import to_categorical # 범주
y = to_categorical(y)
print(y)
print(y.shape) # (150, 3)


x = x.reshape(150, 2, 2, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )






#2. 모델구성
input_01 = Input(shape=(2, 2, 1))
conv2d_01 = Conv2D(filters=64, kernel_size=(1, 1))(input_01)
maxpool_01 = MaxPooling2D(1, 1)(conv2d_01)
dropout_01 = Dropout(0.3)(maxpool_01)
conv2d_02 = Conv2D(filters=64, kernel_size=(1, 1))(dropout_01)
maxpool_02 = MaxPooling2D(1, 1)(conv2d_02)
dropout_02 = Dropout(0.3)(maxpool_02)
flatten_01 = Flatten()(dropout_02)
dense_01 = Dense(100)(flatten_01)
output_01 = Dense(3, activation='softmax')(dense_01)
model = Model(inputs=input_01, outputs=output_01)


import time
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',            # 다중분류의 loss는 $$$당!분!간$$$ categorical_crossentropy 만 쓴다 (20220704)
              optimizer='adam',
              metrics=['accuracy'])
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[earlyStopping], verbose=1) 

#4. 평가, 예측
# loss, acc = model.evaluate(x_test, y_test)
# print('loss : ', loss) 
# print('accuracy : ', acc)                  # 밑에와 같음

results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 


from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print(y_predict)

y_test = np.argmax(y_test, axis=1)
print(y_test)



# r2 = r2_score(y_test, y_predict)
acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)


# loss :  0.015602652914822102  기존훈련
# accuracy :  1.0

# loss :  0.023947110399603844 dropout
# accuracy :  1.0

# CNN모델
# loss :  0.02740670181810856
# accuracy :  1.0
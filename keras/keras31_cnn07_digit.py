from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, MaxPooling2D, Flatten, Conv2D
from sklearn.model_selection import train_test_split        
from sklearn.datasets import load_wine, load_digits                             
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
datasets = load_digits()
x = datasets.data
y = datasets.target 
print(x.shape, y.shape) # (1797, 64) (1797,)    8x8(64)이미지가 1797개 있다는 말  input_dim = 64
print(np.unique(y, return_counts=True))      # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))    이걸 1797,10으로 변환

""" 이미지 보는법 
import matplotlib.pyplot as plt    
plt.gray()
plt.matshow(datasets.images[3])
plt.show()
"""

x = x.reshape(1797, 8, 8, 1)

from tensorflow.keras.utils import to_categorical # 범주
y = to_categorical(y)
print(y)
print(y.shape) # (150, 3)



x_train, x_test, y_train, y_test = train_test_split(x, y,
    test_size=0.33,
    random_state=72
    )




#2. 모델구성
input_01 = Input(shape=(8, 8, 1))
conv2d_01 = Conv2D(filters=64, kernel_size=(2, 2))(input_01)
maxpool_01 = MaxPooling2D(1, 1)(conv2d_01)
dropout_01 = Dropout(0.3)(maxpool_01)
conv2d_02 = Conv2D(filters=64, kernel_size=(1, 1))(dropout_01)
maxpool_02 = MaxPooling2D(1, 1)(conv2d_02)
dropout_02 = Dropout(0.3)(maxpool_02)
flatten_01 = Flatten()(dropout_02)
dense_01 = Dense(100)(flatten_01)
output_01 = Dense(10, activation='softmax')(dense_01)
model = Model(inputs=input_01, outputs=output_01)



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',            
              optimizer='adam',
              metrics=['accuracy'])
                                            
from tensorflow.python.keras.callbacks import EarlyStopping      
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, restore_best_weights=True)          
hist = model.fit(x_train, y_train, epochs=10, batch_size=1, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  # batch_size 디폴트값 32



#4. 평가, 예측
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


# loss :  0.3825773000717163       기존훈련
# accuracy :  0.8888888955116272

# loss :  0.4905156195163727       dropout
# accuracy :  0.8787878751754761


# CNN 모델
# # loss :  1.0110924243927002
# accuracy :  0.9057239294052124
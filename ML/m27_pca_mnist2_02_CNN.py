from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score 
from sklearn.decomposition import PCA
from keras.datasets import mnist
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


(x_train, y_train), (x_test, y_test) = mnist.load_data() # _ = 안 들고오겠다

x = np.append(x_train, x_test, axis=0)
print(x.shape) # (70000, 28, 28)

x = x.reshape(70000, 28*28) # (70000, 784)
print(x.shape)

pca = PCA(n_components=403)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_

cumsum =np.cumsum(pca_EVR)
# print(cumsum)
# print(np.argmax(cumsum >= 0.996) + 1)  

x_train = x[:60000] # (60000, 403)
x_test = x[60000:]  # (10000, 403)

x_train = x_train.reshape(60000, 403, 1, 1)
x_test = x_test.reshape(10000, 403, 1, 1)

from tensorflow.python.keras.utils.np_utils import to_categorical # 범주
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape) # (60000, 10)  



#2. 모델구성
model = Sequential()                                                                             
model.add(Conv2D(filters=64, kernel_size=(1,1), padding='same', input_shape=(403, 1, 1)))
                                              # padding 0이라는 패딩을 씌워서 이미지를 조각낼때 가장자리 부분을 두번 이상 넣어줘서 다른 부분보다 덜 학습되는걸 방지 
                                              # 통상 shape를 다음 레이어에도 유지하고 싶을때 padding을 쓴다                                                                                    
model.add(Conv2D(32, (1, 1), padding='valid', activation='relu'))    # padding='valid' 디폴트
model.add(Conv2D(32, (1, 1), activation='relu'))    
model.add(Conv2D(32, (1, 1), activation='relu'))   
model.add(Flatten())                          # (N, 252)   Flatten을 안써도 하단 dense로 계산된다                                                             
model.add(Dense(32, activation='relu'))    
model.add(Dropout(0.2))                                                                                                     
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint      
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)        
hist = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  
end_time = time.time() - start_time

#4. 결과, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 


from sklearn.metrics import r2_score, accuracy_score 
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)

y_test = tf.argmax(y_test, axis=1)
print(y_test)


acc = accuracy_score(y_test, y_predict)  
print('acc스코어 :', acc)
print('time :', end_time)

# acc스코어 : 0.8799
# time : 41.760329723358154


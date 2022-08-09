from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score 
from sklearn.decomposition import PCA
from keras.datasets import mnist
import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier





# 전 4가지 모델사용
# 784개 DNN으로 만든거(최상의 성능인거 // 0.996이상)과 비교
# time 체크 / fit에서 하고

#1. 나의 최고의 DNN
# time = 69.72170734405518
# acc  = 0.9672

#2. 나의 최고의 CNN
# time = ??
# acc  = ??



#3. PCA 0.95
# time = ??
# acc  = ??

#4. PCA 0.99
# time = ??
# acc  = ??

#5. PCA 0.999
# time = ??
# acc  = ??

#6. PCA 1.0
# time = ??
# acc  = ??



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

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
print(pd.get_dummies(y_train))
print(y_train.shape) # (60000, 10)  



#2. 모델구성
model = Sequential()


# model.add(Dense(64, input_shape=(28*28, ))) - 28x28 사이즈였다는걸 이런식으로 명시도 가능 
model.add(Dense(64, input_shape=(403, )))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
start_time = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint      
earlyStopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)        

  
hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[earlyStopping], verbose=1)  
                                            # batch_size:32 디폴트값 3번정도 말 한듯
end_time = time.time() - start_time


#4. 결과, 예측
results = model.evaluate(x_test, y_test)
print('loss : ', results[0]) 
print('accuracy : ', results[1]) 

y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis=1)
print(y_predict)

y_test = tf.argmax(y_test, axis=1)
print(y_test)

acc = accuracy_score(y_test, y_predict)  
print('accuracy :', acc)
print('time :', end_time)

# acc스코어 : 0.9686


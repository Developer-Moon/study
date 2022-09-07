from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout, Conv2D
import keras
from tensorflow.keras.layers import GlobalAveragePooling2D
from sklearn.metrics import accuracy_score

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. model
activation='relu'
drop=0.2
optimizer='adam'

inputs = Input(shape=(28, 28, 1), name='input')
x = Conv2D(64, (2, 2), padding='valid', activation=activation, name='hidden1')(inputs) # 27, 27, 128
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding='same', activation=activation, name='hidden2')(x)        # 27, 27, 64
# x = Dropout(drop)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), padding='valid', activation=activation, name='hidden3')(x)       # 25, 25, 32
x = Dropout(drop)(x) #
# x = Flatten()(x)     # 25 * 25 * 32 = 20000
x = GlobalAveragePooling2D()(x) # flatten의 너무 많은 연산으로 인해 과적합 방지 GlobalAveragePooling2D게 더 좋은 성능을 낸다는 글이 있다

x = Dense(100, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x) 
outputs = Dense(10, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()



#3. compile
model.compile(optimizer=optimizer, metrics=['acc'], loss='sparse_categorical_crossentropy')
    
import time
start = time.time()
model.fit(x_train, y_train, epochs=20, validation_split=0.4, batch_size=128)
end = time.time() - start


#4.결과예측
loss, acc = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print(y_pred[:10])

y_pred = np.argmax(model.predict(x_test), axis=-1)

print('걸린 시간: ', end)
print('acc score: ', accuracy_score(y_test, y_pred))


# [[1.7922182e-05 4.7359438e-04 4.0449980e-03 4.1827091e-04 3.2535268e-04
#   2.1039566e-03 1.3368359e-06 9.8709178e-01 3.6086653e-06 5.5192919e-03]
#  [1.3878290e-02 5.6718723e-03 9.4934243e-01 4.6238890e-03 5.0043123e-04
#   8.3452584e-03 1.6643722e-02 1.3508995e-04 7.2044122e-04 1.3848805e-04]
#  [3.8051097e-05 9.9878842e-01 1.5450729e-05 1.6716071e-07 5.8987440e-04
#   5.9870454e-06 4.6541909e-04 6.6966066e-05 2.6012049e-05 3.6823574e-06]
#  [9.7063780e-01 3.1398780e-05 1.3221788e-03 1.4422245e-07 6.2434133e-03
#   7.2401926e-06 1.8540129e-02 8.7475355e-06 3.1323307e-03 7.6601500e-05]
#  [3.6105223e-04 9.6903072e-04 9.2263792e-05 1.5822603e-07 9.9529368e-01
#   9.2821381e-07 2.7303354e-04 1.2328499e-03 3.3966060e-06 1.7735846e-03]
#  [1.6419490e-05 9.9947625e-01 1.6124181e-06 1.7938502e-08 3.3063604e-04
#   1.7399168e-06 8.2297192e-05 7.6980723e-05 1.1974050e-05 2.0392577e-06]
#  [8.9311542e-04 5.5614871e-04 5.2278461e-03 3.3076169e-05 9.5085704e-01
#   3.5825826e-04 4.9475017e-03 2.5649376e-03 9.2902972e-04 3.3633094e-02]
#  [1.3352059e-03 1.0024116e-03 7.5432554e-02 2.0031272e-02 1.3087507e-01
#   1.2131216e-03 4.5809746e-03 3.3697870e-03 6.9823779e-02 6.9233584e-01]
#  [3.2104235e-02 3.8957486e-03 9.4235951e-01 2.1482494e-03 9.7820033e-05
#   1.3600630e-02 7.2728458e-04 4.6695108e-03 6.7614521e-05 3.2951226e-04]
#  [8.2026055e-04 3.3555391e-06 2.2298452e-03 1.4031037e-03 3.0921821e-03
#   2.5071241e-03 6.1896797e-05 3.3661053e-02 4.9573340e-02 9.0664786e-01]]
# 걸린 시간:  87.64878630638123
# acc score:  0.9294

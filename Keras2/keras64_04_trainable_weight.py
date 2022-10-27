from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np


#1. Data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])


#2. Model
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
#
print(model.weights)
print("==================================================================================")
print(model.trainable_weights)
print("==================================================================================")
print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 6

model.trainable=False               # 전이학습할때 사용

print(len(model.weights))           # 6
print(len(model.trainable_weights)) # 0

print("==================================================================================")
print(model.trainable_weights)      # []

model.summary()

model.compile(loss='mse', optimizer="adam")

model.fit(x, y, batch_size=1, epochs=100)

y_predict = model.predict(x)
print(y_predict[:3])
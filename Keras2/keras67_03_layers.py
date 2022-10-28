from keras.models import Sequential
from keras.layers import Dense
 

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))


#1.
# model.trainable=False
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17

#2.
# for layer in model.layers :
#     layer.trainable = False
# Total params: 17
# Trainable params: 0
# Non-trainable params: 17    
    
model.layers[0].trainable = False # Dense
# Total params: 17
# Trainable params: 11
# Non-trainable params: 6


# model.layers[1].trainable = False # Dense_1
# Total params: 17
# Trainable params: 9
# Non-trainable params: 8


# model.layers[2].trainable = False # Dense_2
# Total params: 17
# Trainable params: 14
# Non-trainable params: 3

model.summary()

print(model.layers)
# [<keras.layers.core.dense.Dense object at 0x000002A2E493CFA0>,
#  <keras.layers.core.dense.Dense object at 0x000002A2F1014D30>,
#  <keras.layers.core.dense.Dense object at 0x000002A2F1014D90>]
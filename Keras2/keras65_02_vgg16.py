from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16


# model = VGG16()
# model = VGG16() # include_top=True, input_shape=(224, 224, 3)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# vgg16.trainable=False # 가중치 동결
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

model.trainable = False

model.summary()
                                    # Trainable:True / VGG False / model False
print(len(model.weights))           # 30             / 30        / 30           all weight 개수
print(len(model.trainable_weights)) # 30             / 4         / 0            훈련 weight 개수
                                    #                 FC만 훈련









# ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data,
# or specify an `input_shape` argument in the first layer(s) for automatic build.

# 버전문제
# from keras.models import Sequential
# from keras.layers import Dense, Flatten
# from keras.applications import VGG16, VGG19
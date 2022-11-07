# Question
#
# Create a classifier for the CIFAR10 dataset
# Note that the test will expect it to classify 10 classes and that the input shape should be
# the native CIFAR size which is 32x32 pixels with 3 bytes color depth

import tensorflow as tf

def solution_model():
    cifar = tf.keras.datasets.cifar10

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
    from keras.utils import to_categorical
    from sklearn.metrics import accuracy_score
    import numpy as np
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=(32, 32, 3),activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, epochs=50, verbose=1)

    loss = model.evaluate(x_test, y_test)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    print('loss : ', loss)
    print('pred :', y_pred)
    print('ACC:', accuracy_score(y_test, y_pred))

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
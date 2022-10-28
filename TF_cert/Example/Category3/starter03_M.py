# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/testdata/')
    zip_ref.close()

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=5,
        rotation_range=5,
        zoom_range=1.2,
        shear_range=0.7,
        fill_mode='nearest'
        )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=5,
        rotation_range=5,
        zoom_range=1.2,
        shear_range=0.7,
        fill_mode='nearest'
        )

    train_generator = train_datagen.flow_from_directory(
        'tmp/horse-or-human/',
        target_size=(300, 300),
        batch_size=500,
        class_mode='binary',
        shuffle=True
        )

    validation_generator = validation_datagen.flow_from_directory(
        'tmp/testdata/',
        target_size=(300, 300),
        batch_size=500,
        class_mode='binary',
        shuffle=True
        )

    scale_datagen = ImageDataGenerator(rescale=1./255)
    
    x_train = train_generator[0][0]
    y_train = train_generator[0][1]
    x_test = validation_generator[0][0]
    y_test = validation_generator[0][1]
    
    augument_size = 500
    randidx = np.random.randint(x_train.shape[0], size=augument_size)
    
    x_augument = x_train[randidx].copy()
    y_augument = y_train[randidx].copy()
    
    x_augument = train_datagen.flow(x_augument, y_augument, batch_size=augument_size, shuffle=False).next()[0]

    x_train = scale_datagen.flow(x_train, y_train, batch_size=augument_size, shuffle=False).next()[0]

    
    x_train = np.concatenate((x_train, x_augument))
    y_train = np.concatenate((y_train, y_augument))

    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here
        tf.keras.layers.Conv2D(64, (2,2), input_shape=(300,300,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

    
    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(x_train, y_train, epochs=100, batch_size=10)
    
    y_predict = model.predict(x_test)
    img_test_predict = model.predict(img_test)

    y_predict = tf.argmax(y_predict, axis=1)
    y_test = tf.argmax(y_test, axis=1)
    img_test_predict = tf.argmax(img_test_predict, axis=1)

    # : If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
# Question
#
# Create a classifier for the CIFAR10 dataset
# Note that the test will expect it to classify 10 classes and that the input shape should be
# the native CIFAR size which is 32x32 pixels with 3 bytes color depth

import tensorflow as tf

def solution_model():
    cifar = tf.keras.datasets.cifar10

    # YOUR CODE HERE
    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
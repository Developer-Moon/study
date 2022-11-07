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
# Computer Vision with CNNs
#
# For this exercise, build and train a cats v dogs classifier
# using the Cats v Dogs dataset from TFDS.
# Be sure to use the final layer as shown 
#     (Dense, 2 neurons, softmax activation)
#
# The testing infrastructure will resize all images to 224x224
# with 3 bytes of color depth. Make sure your input layer trains
# images to that specification, or the tests will fail.
#
# Make sure your output layer is exactly as specified here, or the 
# tests will fail.
#
# HINT: This is a large dataset and might take a long time to train.
# When experimenting with your architecture, use the splits API to train
# on a smaller set, and then gradually increase the training set size until
# it works very well. This is trainable in reasonable time, even on a CPU
# if architected correctly.
# NOTE: The dataset has some corrupt JPEG data in the images. If you see warnings
# about extraneous bytes before marker 0xd9, you can ignore them safely


import tensorflow_datasets as tfds
import tensorflow as tf


dataset_name = 'cats_vs_dogs'
dataset, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)

def preprocess(features):
    # YOUR CODE HERE

def solution_model():
    train_dataset = dataset.map(preprocess).batch(32)

    model = # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 1 NEURON ACTIVATED BY SIGMOID
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

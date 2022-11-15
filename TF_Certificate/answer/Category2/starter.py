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
# Basic Datasets question
#
# For this task you will train a classifier for Iris flowers using the Iris dataset
# The final layer in your neural network should look like: tf.keras.layers.Dense(3, activation=tf.nn.softmax)
# The input layer will expect data in the shape (4,)
# We've given you some starter code for preprocessing the data
# You'll need to implement the preprocess function for data.map

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = tfds.load('iris', split='train')
valid_data = tfds.load('iris', split='train[80%:]')


def preprocess(features):
    # YOUR CODE HERE
    # Should return features and one-hot encoded labels
    feature, label = features['features'], tf.one_hot(features['label'], 3)
    return feature, label

def solution_model():
    train_dataset = data.map(preprocess).batch(10)
    # valid_dataset = valid_data.map(preprocess).batch(10)

    model = Sequential([
        Dense(512, activation='relu'),ss
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax'),
    ])


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(train_dataset, epochs=100, validation_freq=0.2)


    # YOUR CODE TO TRAIN A MODEL HERE
    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")


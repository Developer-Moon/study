# ==============================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative to
# its difficulty. So your Category 1 question will score significantly less
# than your Category 5 question.
#
# WARNING: Do not use lambda layers in your model, they are not supported
# on the grading infrastructure. You do not need them to solve the question.
#
# WARNING: If you are using the GRU layer, it is advised not to use the
# recurrent_dropout argument (you can alternatively set it to 0),
# since it has not been implemented in the cuDNN kernel and may
# result in much longer training times.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ==============================================================================
# TIME SERIES QUESTION
#
# Build and train a neural network to predict weather time series.
# Using a window of past 40 observations, train the model to
# predict the next one observation.
# ==============================================================================
#
# ABOUT THE DATASET
#
# This is a custom dataset created by Google for the purpose of this
# examination.
# The dataset consists of temperature values ordered by time.
# =============================================================================
# INSTRUCTIONS
#
# Complete the code in the following functions:
# 1. windowed_dataset()
# 2. solution_model()
#
# Your code will fail to be graded if the following criteria are not met:
#
# 1. Model input shape must be [BATCH_SIZE, N_PAST = 40, 1], since the
#    testing infrastructure expects a window of past N_PAST = 40 observations
#    of the variable to predict the next observation of the variable.
#
# 2. Model output_shape must be [BATCH_SIZE, N_FEATURES = 1]
#    Refer to the code to see the definitions for these constants.
#
# 3. The last layer of your model must be a Dense layer with 1 neuron since
#    the model is expected to predict observations of 1 feature.
#
# 4. Don't change the values of the following constants:
#    SPLIT_TIME, N_FEATURES, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT, in
#    solution_model() (See code for additional note on BATCH_SIZE).
#
# 5. Code for normalizing the data is provided - don't change it.
#    Changing the normalizing code will affect your score.
#
# HINT: Your neural network must have a validation MAE of approximately 0.3 or
# less on the normalized dataset for top marks.

import zipfile
import tensorflow as tf
import numpy as np
import urllib


# This function downloads and extracts the dataset to the directory that
# contains this file.
# DO NOT CHANGE THIS CODE
# (unless you need to change https to http)
def download_and_extract_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/certificate/weather_station.zip'
    urllib.request.urlretrieve(url, 'weather_station.zip')
    with zipfile.ZipFile('weather_station.zip', 'r') as zip_ref:
        zip_ref.extractall()


# This function is used to load the time series data from a
# csv file "station.csv". Each line has 12 comma separated observations
# corresponding to months in a year. The first line in the csv is the header
# having names of columns(months).
# The function reads the CSV line by line and appends observations for
# each month in a year, to a 1D array named temperatures so as to record
# monthly data for temperatures as the dataset.
def get_data():
    data_file = "station.csv"
    f = open(data_file)
    data = f.read()
    f.close()
    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    temperatures = []
    for line in lines:
        if line:
            linedata = line.split(',')
            linedata = linedata[1:13]
            for item in linedata:
                if item:
                    temperatures.append(float(item))

    series = np.asarray(temperatures)
    time = np.arange(len(temperatures), dtype="float32")
    return time, series


# This function is used to map the time series dataset into windows of
# features and respective targets, to prepare it for training and
# validation. First element of the first window will be the first element of
# the dataset. Consecutive windows are constructed by shifting
# the starting position of the first window forward, one at a time (indicated
# by shift=1). For a window of n_past number of observations of the time
# indexed variable in the dataset, the target for the window is the next
# n_future number of observations of the variable, after the end of the
# window.

# COMPLETE THE CODE IN THE FOLLOWING FUNCTION.
def windowed_dataset(series, batch_size, n_past=40, n_future=1, shift=1):
    # Adds an extra dimension of size 1 to the series.
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)

    # This line converts the dataset into a windowed dataset where a
    # window consists of both the observations to be included as
    # features and the targets.
    #
    # DON'T change the shift parameter. The test windows are
    # created with the specified shift and hence it might affect your
    # scores. You must calculate the window size so that based on
    # the past 40 observations (observations at time steps t=1,t=2,
    # ...t=40) of the 1 variable in the dataset, you predict the next 1
    # observation (observation at time step t=41) of the 1 variable in the
    # dataset.

    # Hint: Each window should include both the past observations and the
    # future observations which are to be predicted. Calculate the window size
    # based on n_past and n_future.
    # Note: This line returns a Dataset of Datasets (each dataset holds
    # elements of one window).
    ds = ds.window(size=  # YOUR CODE HERE,
                   shift = shift,
                           drop_remainder = True)

    # This takes the Dataset of Datasets and flattens it into a single
    # dataset of tensors.
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))

    # Now, each element of the dataset is a tensor holding
    # n_past + n_future observations corresponding to each window.
    # This line maps each window of the dataset to the form
    # (n_past observations, n_future observations) which is the input format
    # needed for training the model.
    # Hint: Use a lambda function to map each window in the dataset to its
    # respective (features, targets).
    ds = ds.map(
        # YOUR CODE HERE
    )

    ds = ds.batch(batch_size).prefetch(1)

    return ds


# This function downloads the dataset, loads the data from CSV file,
# normalizes the data and splits the dataset into train and validation
# sets. It also uses windowed_dataset() to split the data into
# windows of observations and targets. (Refer to the function for more
# information). Finally it defines, compiles and trains a neural network. This
# function returns the final trained model.

# COMPLETE THE CODE IN THIS FUNCTION
def solution_model():
    # DO NOT CHANGE THIS CODE
    # Loads the data and reads it line by line to extract time ordered values
    # of the single feature.
    download_and_extract_data()
    time, series = get_data()

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    mean = series.mean(axis=0)
    series -= mean
    std = series.std(axis=0)
    series /= std

    # The data is split into training and validation sets at SPLIT_TIME.
    SPLIT_TIME = 780  # DO NOT CHANGE THIS CODE
    x_train = series[:SPLIT_TIME]
    x_valid = series[SPLIT_TIME:]

    # DO NOT CHANGE THIS CODE
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    # There is only one feature varying with time in this dataset.
    # We predict one feature using past observations of same feature.
    N_FEATURES = 1 # DO NOT CHANGE THIS CODE

    # DO NOT CHANGE BATCH_SIZE IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    # TEST WILL FAIL TO GRADE YOUR SCORE IN SUCH CASES.
    # In other cases, it is advised not to change the batch size since it
    # might affect your final scores. While setting it to a lower size
    # might not do any harm, setting it to higher sizes might affect your
    # scores.
    BATCH_SIZE = 20  # ADVISED NOT TO CHANGE THIS

    # DO NOT CHANGE N_PAST, N_FUTURE, SHIFT. The tests will fail to run
    # on the server.
    # Number of past time steps based on which future observations should
    # be predicted
    N_PAST = 40  # DO NOT CHANGE THIS

    # Number of future time steps which are to be predicted.
    N_FUTURE = 1  # DO NOT CHANGE THIS

    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1  # DO NOT CHANGE THIS

    # Code to create windowed train and validation datasets.
    train_dataset = windowed_dataset(series=x_train,
                                     batch_size=BATCH_SIZE,
                                     n_past=N_PAST,
                                     n_future=N_FUTURE)
    valid_dataset = windowed_dataset(series=x_valid,
                                     batch_size=BATCH_SIZE,
                                     n_past=N_PAST,
                                     n_future=N_FUTURE)

    model = tf.keras.models.Sequential([

        # ADD YOUR LAYERS HERE.

        # If you don't follow the instructions in the following comments,
        # tests will fail to grade your code:
        # The input layer of your model must have an input shape of
        # (BATCH_SIZE,N_PAST=40,1)
        # The model must have an output shape of (BATCH_SIZE, 1).
        # Make sure that there are N_FEATURES = 1 neurons in the final dense
        # layer since model predicts one feature.

        # WARNING: If you are using the GRU layer, it is advised not to use the
        # recurrent_dropout argument (you can alternatively set it to 0),
        # since it has not been implemented in the cuDNN kernel and may
        # result in much longer training times.
        tf.keras.layers.Dense(N_FEATURES)
    ])

    # Code to train and compile the model
    optimizer =  # YOUR CODE HERE
    model.compile(
        # YOUR CODE HERE
    )
    model.fit(
        # YOUR CODE HERE
    )

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("c5q2.h5")

'''
File: \run.py
Project: 3- SimpleAutoEncoder
Created Date: Saturday March 10th 2018
Author: Huisama
-----
Last Modified: Saturday March 10th 2018 2:58:37 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import AdditiveGaussianNoiseAutoEncoder as AGN

import sklearn.preprocessing as prep
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets("../0-Dataset/MNIST", one_hot=True)

'''
    standard_scale scale the sample's future as N(0,1)
'''
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]


X_train, X_test = standard_scale(MNIST.train.images, MNIST.test.images)

# Define constants
N_SAMPLES = int(MNIST.train.num_examples)
EPOCHS = 100
BATCH_SIZE = 64
DISPLAY_STEP = 1

# Get the AGN object
autoencoder = AGN.AdditiveGaussianNoiseAutoEncoder(
    n_input=784, n_hidden=400,
    transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
    scale=0.01)

# Do training
for epoch in range(EPOCHS):
    avg_cost = 0.
    total_batch = int(N_SAMPLES / BATCH_SIZE)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, BATCH_SIZE)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / N_SAMPLES * BATCH_SIZE

    if epoch % DISPLAY_STEP == 0:
        print("Epoch:", '%04d' % (epoch + 1),
              "cost=", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
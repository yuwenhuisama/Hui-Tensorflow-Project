'''
File: \BidirectinalLSTMClassifier.py
Project: 13-BidirectinalLSTMClassifier
Created Date: Thursday March 22nd 2018
Author: Huisama
-----
Last Modified: Thursday March 22nd 2018 9:19:51 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets('../0-Dataset/MNIST', one_hot = True)

LEARNING_RATE = 0.01
MAX_SAMPLES = 400000
BATCH_SIZE = 128
DISPLAY_STEP = 10

N_INPUT =  28
N_STEPS = 28
N_HIDDEN = 256
N_CLASSES = 10

x = tf.placeholder('float', [None, N_STEPS, N_INPUT])
y = tf.placeholder('float', [None, N_CLASSES])

# 2 for forward and backward RNN layers
weights = tf.Variable(tf.random_normal([2 * N_HIDDEN, N_CLASSES]))
biases = tf.Variable(tf.random_normal([N_CLASSES]))

def BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, N_INPUT])
    x = tf.split(x, N_STEPS)

    # x will be like: 
    # 
    # [
    #    (row1)
    #   [
    #     [
    #        row value of row in each batch
    #     ]
    #   ]
    #    (row2)
    #   [
    #     [
    #        row value of row2 in each batch
    #     ]
    #   ]
    # ]

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN, forget_bias = 1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN, forget_bias = 1.0)

    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)

    return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))

optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

# Do training
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * BATCH_SIZE < MAX_SAMPLES:
        batch_x, batch_y = MNIST.train.next_batch(BATCH_SIZE)
        # reshape (batch_size, width * height) to (batch_size, width, height)
        batch_x = batch_x.reshape((BATCH_SIZE, N_STEPS, N_INPUT))
        sess.run(optimizer, feed_dict = { x: batch_x, y: batch_y })

        if step % DISPLAY_STEP:
            acc = sess.run(accuracy, feed_dict = { x: batch_x, y: batch_y })
            loss = sess.run(cost, feed_dict = { x: batch_x, y : batch_y})
            print("Iter " + str(step * BATCH_SIZE) + ', Minibatch Loss= ' + '{:.6f}'.format(loss) + ', Train Accuracy= ' + '{:.5f}'.format(acc))
        step += 1
    print('Optimization Finished!')

    # Do testing
    test_len = 10000
    test_data = MNIST.test.images[:test_len].reshape((-1, N_STEPS, N_INPUT))
    test_label = MNIST.test.labels[:test_len]
    print('Testing Accuracy:', sess.run(accuracy, feed_dict = {
        x: test_data,
        y: test_label
    }))
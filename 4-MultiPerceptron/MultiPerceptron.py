'''
File: \MultiPerceptron.py
Project: 4-MultiPerceptron
Created Date: Saturday March 10th 2018
Author: Huisama
-----
Last Modified: Saturday March 10th 2018 8:21:26 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Dataset
MNIST = input_data.read_data_sets('../0-Dataset/MNIST', one_hot = True)
SESS = tf.InteractiveSession()

IN_UNITS = 784
H1_UNITS = 300

# Parameters
# Initialize W1 ~ N(any, 0.1^2)
W1 = tf.Variable(tf.truncated_normal([IN_UNITS, H1_UNITS], stddev = 0.1))
b1 = tf.Variable(tf.zeros([H1_UNITS]))
W2 = tf.Variable(tf.zeros([H1_UNITS, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, IN_UNITS])
keep_prob = tf.placeholder(tf.float32)

# Define structure of NN
# Input -> Hidden layer -> Dropout layer -> Output
# Input to hidden layer, using relu activate function
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# Link a dropout layer after hidden layer
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

# Dropout hidden layer to output
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# Define loss function
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# Begin training
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = MNIST.train.next_batch(100)
    train_step.run({
        x: batch_xs,
        y_: batch_ys,
        keep_prob: 0.75,
    })

# Adjust model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({
    x: MNIST.test.images,
    y_: MNIST.test.labels,
    keep_prob: 1.0
}))

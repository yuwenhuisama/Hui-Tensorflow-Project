'''
File: \softmax.py
Project: 2-Softmax-Regression
Created Date: Saturday March 10th 2018
Author: Huisama
-----
Last Modified: Saturday March 10th 2018 11:34:48 am
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Dataset
MNIST = input_data.read_data_sets("../0-Dataset/MNIST", one_hot=True)

# NN defination
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784], "x")
# 784 dim input, 10 dim output
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Loss function defination
y_ = tf.placeholder(tf.float32, [None, 10], "y_")
# Get the mean of cross entropy every batch
cross_entrypy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Training setting
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entrypy)

# Begin training
tf.global_variables_initializer().run()

# Iteration for training, batch by batch
for i in range(1000):
    batch_xs, batch_ys = MNIST.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

# Test training result
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: MNIST.test.images, y_: MNIST.test.labels}))

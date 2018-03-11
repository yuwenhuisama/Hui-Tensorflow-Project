'''
File: \SimpleCnn.py
Project: 5-SimpleCNN
Created Date: Sunday March 11th 2018
Author: Huisama
-----
Last Modified: Sunday March 11th 2018 1:05:33 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

MNIST = input_data.read_data_sets("../0-Dataset/MNIST", one_hot = True)
SESS = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

'''
    Set bias to 0.1 (small value) to avoid dead neurous
'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

'''
    max_pool_2x2 is used to undersampling an 2x2 block to 1x1 block in the image
'''
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# Reshape 1D vector to 2D matrix for the image data read.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Define NN structure
# Convolution layer 1
# 32 convolution kernel of 5x5 for 1 channel
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = weight_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolution layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# FCN layer connected to Convolution layer 2
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = weight_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Add a Dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Begin Train
tf.global_variables_initializer().run()
for i in range(20000):
    batch = MNIST.train.next_batch(50)
    if (i % 100 == 0):
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0],
            y_: batch[1],
            keep_prob: 1.0,
        })
    print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict = {
        x: batch[0],
        y_: batch[1],
        keep_prob: 0.5,
    })

print("test accuracy %g" % accuracy.eval(feed_dict = {
    x: MNIST.test.images,
    y_: MNIST.test.labels,
    keep_prob: 1.0,
}))

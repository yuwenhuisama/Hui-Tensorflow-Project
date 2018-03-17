'''
File: \VGGNet.py
Project: 8-VGGNet
Created Date: Saturday March 17th 2018
Author: Huisama
-----
Last Modified: Saturday March 17th 2018 4:15:02 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

from datetime import datetime
import math
import time
import tensorflow as tf

NUM_BATCHS = 100
BATCH_SIZE = 32

'''
    Create a convolution layer
'''
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w", shape = [kh, kw, n_in, n_out], dtype = tf.float32,
                                    initializer = tf.contrib.layers.xavier_initializer_conv2d())
        
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape = [n_out], dtype = tf.float32)
        biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name = scope)
        p += [kernel, biases]
        return activation

'''
    Create a full connection layer
''' 
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        weight = tf.get_variable(scope + 'w', shape = [n_in, n_out], dtype = tf.float32,
                                    initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32))

        activation = tf.nn.relu_layer(input_op, weight, biases, name = scope)
        p += [weight, biases]
        return activation

'''
    Create a max pool layer
'''
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize = [1, kh, kw, 1],
                            strides = [1, dh, dw, 1], padding = 'SAME', name = name)

'''
    Define VGGNet
'''
def inference_op(input_op, keep_prob):
    p = []

    # 1st group of convolution layers
    conv1_1 = conv_op(input_op, name = 'conv1_1', kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
    conv1_2 = conv_op(conv1_1, name = 'conv1_2', kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
    pool1 = mpool_op(conv1_2, name = 'pool1', kh = 2, kw = 2, dw = 2, dh = 2)

    # 2nd group of convolution layers
    conv2_1 = conv_op(pool1, name = 'conv2_1', kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)
    conv2_2 = conv_op(conv2_1, name = 'conv2_2', kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)
    pool2 = mpool_op(conv2_2, name = 'pool2', kh = 2, kw = 2, dh = 2, dw = 2)

    # 3rd group of convolution layers
    conv3_1 = conv_op(pool2, name = 'conv3_1', kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)
    conv3_2 = conv_op(conv3_1, name = 'conv3_2', kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)
    conv3_3 = conv_op(conv3_2, name = 'conv3_3', kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)
    pool3 = mpool_op(conv3_3, name = 'pool3', kh = 2, kw = 2, dh = 2, dw = 2)

    # 4th group of convolution layers
    conv4_1 = conv_op(pool3, name = 'conv4_1', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv4_2 = conv_op(conv4_1, name = 'conv4_2', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv4_3 = conv_op(conv4_2, name = 'conv4_3', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    pool4 = mpool_op(conv4_3, name = 'pool4', kh = 2, kw = 2, dh = 2, dw = 2)

    # 5th group of convolution layers
    conv5_1 = conv_op(pool4, name = 'conv5_1', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv5_2 = conv_op(conv5_1, name = 'conv5_2', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv5_3 = conv_op(conv5_2, name = 'conv5_3', kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    pool5 = mpool_op(conv5_3, name = 'pool5', kh = 2, kw = 2, dh = 2, dw = 2)

    # Define full connection layers
    shape = pool5.get_shape()
    # flattend_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [shape[0].value, -1], name = 'resh1')

    fc6 = fc_op(resh1, name = 'fc6', n_out = 4096, p = p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name = 'fc6_drop')

    fc7 = fc_op(fc6_drop, name = 'fc7', n_out = 4096, p = p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

    # Output layer
    fc8 = fc_op(fc7_drop, name = 'fc8', n_out = 1000, p = p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p

def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_druation_squared = 0.0
    for i in range(NUM_BATCHS + num_steps_burn_in):
        start_time = time.time()
        session.run(target, feed_dict = feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_druation_squared += duration * duration
    mn = total_duration / NUM_BATCHS
    vr = total_druation_squared / NUM_BATCHS - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, NUM_BATCHS, mn, sd))

def run_betchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([
            BATCH_SIZE,
            image_size,
            image_size,
            3
        ], dtype = tf.float32, stddev = 1e-1))

        keep_prob = tf.placeholder(tf.float32)
        predictions, _, fc8, p = inference_op(images, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # Forward learning goes to the end of the NN
        time_tensorflow_run(sess, predictions, { keep_prob: 1.0 }, "Forward")
        # Backward learning begin from fc8
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, { keep_prob: 0.5}, "Backward")

run_betchmark()

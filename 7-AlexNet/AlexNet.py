'''
File: \AlexNet.py
Project: 7-AlexNet
Created Date: Friday March 16th 2018
Author: Huisama
-----
Last Modified: Saturday March 17th 2018 3:22:27 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

from datetime import datetime
import math
import time
import tensorflow as tf
import numpy as np

BATCH_SIZE = 32
NUM_BATCHES = 100

'''
    Xavier initializer
'''
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

'''
    Print layer tensor's size
'''
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

'''
    Generate AlexNet-like NN
'''
def inference(images):
    parameters = []
    
    with tf.name_scope('conv1') as scope:
        # images ** kernel + biases
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = 'biases')
        # The last dimension equals to the number of convolution kernels
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name = scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')
    print_activations(pool1)

    with tf.name_scope('conv_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [192], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9, beta = 0.75, name = 'lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2')
    print_activations(pool2)

    with tf.name_scope('conv_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [384], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    with tf.name_scope('conv_4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    with tf.name_scope('conv_5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool5')
    print_activations(pool5)

    with tf.name_scope('fcn_1') as scope:
        s1 = pool5.get_shape()
        flat_pool5 = tf.reshape(pool5, [s1[0].value, -1])
        W1 = tf.Variable(xavier_init(6*6*256, 4096))
        b1 = tf.Variable(tf.zeros([4096]), dtype=tf.float32)
        fcn1 = tf.add(tf.matmul(flat_pool5, W1), b1)
        print_activations(fcn1)
    
    with tf.name_scope('fcn_2') as scope:
        W2 = tf.Variable(xavier_init(4096, 1000))
        b2 = tf.Variable(tf.zeros([1000]), dtype=tf.float32)
        fcn2 = tf.add(tf.matmul(fcn1, W2), b2)
        print_activations(fcn2)

    return fcn2, parameters

def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(NUM_BATCHES + num_steps_burn_in):
        start_time = time.time()
        session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / NUM_BATCHES
    vr = total_duration_squared / NUM_BATCHES - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec/ batch' % (datetime.now(), info_string, NUM_BATCHES, mn, sd))

'''
    Just test the learning speed for forward-learning and backward-learning every batch, because ImageNet dataset is too big ╮(╯▽╰)╭
'''
def run_benchmark():
    # Radom image data
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([
            BATCH_SIZE,
            image_size,
            image_size,
            3,
        ], dtype = tf.float32, stddev = 1e-1))

        fcn, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, fcn, 'Forward')

        '''
            Virtual loss function
        '''
        objective = tf.nn.l2_loss(fcn)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Backward")

run_benchmark()

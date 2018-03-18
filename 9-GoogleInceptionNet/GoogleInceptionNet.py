'''
File: \GoogleInceptionNet.py
Project: 9-GoogleInceptionNet
Created Date: Saturday March 17th 2018
Author: Huisama
-----
Last Modified: Sunday March 18th 2018 11:25:29 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import tensorflow as tf
slim = tf.contrib.slim

def trunc_normal(stddev): return tf.truncated_normal_initializer(0.0, stddev)

def iniception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, batch_norm_var_collection='moving_vars'):
    batch_norm_params = {
        'decay': 0.9997,
        'epsilon': 0.001,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection]
        }
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params) as sc:
            return sc

'''
    Define Inception NN
'''
def inception_v3_base(inputs, scope=None):
    end_points = {}

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='VALID'):
            # 299 x 299 x 3
            net = slim.conv2d(inputs, 32, [3, 3],
                              stride=2, scope='Conv2d_1a_3x3')
            # 149 x 149 x 32
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
            # 147 x 147 x 32
            net = slim.conv2d(
                net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
            # 147 x 147 x 64
            net = slim.max_pool2d(
                net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            # 73 x 73 x 64
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
            # 73 x 73 x 80.
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            # 71 x 71 x 192.
            net = slim.max_pool2d(
                net, [3, 3], stride=2, scope='MaxPool_5a_3x3')
            # 35 x 35 x 192.

        # Inception blocks
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # mixed: 35 x 35 x 256.
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(
                        branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_1: 35 x 35 x 288.
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(
                        branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_2: 35 x 35 x 288.
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(
                        branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_3: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)

            # mixed4: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(
                        branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(
                        branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(
                        branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(
                        branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_5: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(
                        branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(
                        branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(
                        branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(
                        branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_6: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(
                        branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(
                        branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(
                        branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(
                        branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_7: 17 x 17 x 768.
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 192, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(
                        branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(
                        branch_2, 192, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(
                        branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(
                        branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            # Used to adjust final result (Auxiliary Classifier)
            end_points['Mixed_6e'] = net

            # mixed_8: 8 x 8 x 1280.
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(
                        branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(
                        branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            # mixed_9: 8 x 8 x 2048.
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [
                                    1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [
                                    1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # mixed_10: 8 x 8 x 2048.
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(
                        net, 320, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(
                        net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [
                                    1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(
                        net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(
                        branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [
                                    1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(
                        net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(
                        branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            return net, end_points

def inception_v3(inputs, 
    num_classes = 1000, is_training = True, dropout_keep_prob = 0.8, prediction_fn = slim.softmax,
    spatial_squeeze = True, reuse = None, scope = 'InceptionV3'):
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse = reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training = is_training):
            net, end_points = inception_v3_base(inputs, scope = scope)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):
            aux_logits = end_points['Mixed_6e']
            with tf.variable_scope('AuxLogits'):
                aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride = 3, padding = 'VALID', scope = 'AvgPool_1a_5x5')
                aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope = 'Conv2d_1b_1x1')
                aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer = trunc_normal(0.01), padding = 'VALID', scope = 'Conv2d_2a_5x5')
                aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn = None, normalizer_fn = None,
                                            weights_initializer = trunc_normal(0.001), scope = 'Conv2d_2b_1x1')
                if spatial_squeeze:
                    aux_logits = tf.squeeze(aux_logits, [1, 2], name = 'SpatialSqueeze')
                end_points['AuxLogits'] = aux_logits
            with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, [8, 8], padding = 'VALID', scope = 'AvgPool_1a_8x8')
                net = slim.dropout(net, keep_prob = dropout_keep_prob, scope = 'Dropout_1b')
                end_points['Prelogits'] = net
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'Conv2d_1c_1x1')

                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name = 'SpatialSqueeze')
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope = 'Predictions')
    return logits, end_points 

BATCH_SIZE = 32
NUM_BATCHES = 100
HEIGHT, WIDTH = 299, 299

from datetime import datetime
import math
import time
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
                print ('%s: step %d, duration = %.3f' %
                       (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / NUM_BATCHES
    vr = total_duration_squared / NUM_BATCHES - mn * mn
    sd = math.sqrt(vr)
    print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
           (datetime.now(), info_string, NUM_BATCHES, mn, sd))

def run_benchmark():
    # Radom image data
    with tf.Graph().as_default():
        inputs = tf.random_uniform((BATCH_SIZE, HEIGHT, WIDTH, 3))
        with slim.arg_scope(iniception_v3_arg_scope()):
            logits, _ = inception_v3(inputs, is_training = False)

        random_labels = tf.random_uniform((BATCH_SIZE, 1000))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = random_labels)
        optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, logits, "Forward")

        # objective1 = tf.nn.l2_loss(end_points['AuxLogits'])
        # grad1 = tf.gradients(objective1)
        # objective2 = tf.nn.l2_loss(logits)
        # grad2 = tf.gradients(objective2)

        time_tensorflow_run(sess, optimizer, "Backward")

run_benchmark()
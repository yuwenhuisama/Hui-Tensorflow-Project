'''
File: \ResNet.py
Project: 10-ResNet
Created Date: Monday March 19th 2018
Author: Huisama
-----
Last Modified: Monday March 19th 2018 9:41:29 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import collections
import tensorflow as tf
slim = tf.contrib.slim

'''
    Define ResNet block
'''
class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A nammed tuple describing a ResNet block'

'''
    Do subsample with designated stride
'''
def subsample(inputs, factor, scope = None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride = factor, scope = scope)

'''
    Do 2d convolution
'''
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope = None):
    if stride == -1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = 1, padding = 'SAME', scope = scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride = stride, padding = 'VALID', scope = scope)

'''
    Connect ResNet block to net
'''
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections = None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values = [net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, depth = unit_depth, depth_bottleneck = unit_depth_bottleneck, stride = unit_stride)
                    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net

def resnet_arg_scope(is_training = True, weight_decay = 0.0001, batch_norm_decay = 0.997, batch_norm_epsilon = 1e-5, batch_norm_scale = True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope([slim.conv2d],
            weights_regularizer = slim.l2_regularizer(weight_decay),
            weights_initializer = tf.variance_scaling_initializer(),
            activation_fn = tf.nn.relu,
            normalizer_fn = slim.batch_norm,
            normalizer_params = batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding = 'SAME') as arg_sc:
                return arg_sc

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections = None, scope = None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank = 4)
        preact = slim.batch_norm(inputs, activation_fn = tf.nn.relu, scope = 'preact')
        # If the depth of input equals to the depth designated, then do subsampling by stride
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        # Otherwise to use convolution to adjust the number of channels of inputs and do subsampling by stride (1x1 convolution can also act as subsampling)
        else:
            shortcut = slim.conv2d(preact,
                depth, [1, 1], stride = stride, normalizer_fn = None,
                activation_fn = None, scope = 'shortcut')
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride = 1, scope = 'conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope = 'conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride = 1, normalizer_fn = None, activation_fn = None, scope = 'conv3')

        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

'''
    Define ReseNet NN structure
'''
def resnet_v2(inputs, blocks, num_classes = None, global_pool = True, include_root_block = True, reuse = None, scope = None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse = reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections = end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn = None, normalizer_fn = None):
                    net = conv2d_same(net, 64, 7, stride = 2, scope = 'conv1')
                net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'pool1')
            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn = tf.nn.relu, scope = 'postnorm')
            if global_pool:
                # mean all the feature map's tensor value (demension 1 and 2 with beginning index 0)
                net = tf.reduce_mean(net, [1, 2], name = 'pool5', keep_dims = True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predictions'] = slim.softmax(net, scope = 'predictions')
            return net, end_points_collection

def resnet_v2_50(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

def resnet_v2_101(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 22 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

def resnet_v2_152(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)]* 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

def resnet_v2_200(inputs, num_classes = None, global_pool = True, reuse = None, scope = 'resnet_v2_101'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 23 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block = True, reuse = reuse, scope = scope)

BATCH_SIZE = 32
NUM_BATCHES = 100
HEIGHT, WIDTH = 224, 224

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
    # Random image data
    with tf.Graph().as_default():
        inputs = tf.random_uniform((BATCH_SIZE, HEIGHT, WIDTH, 3))
        with slim.arg_scope(resnet_arg_scope(is_training = False)):
            net, _ = resnet_v2_152(inputs, 1000)

        random_labels = tf.random_uniform((BATCH_SIZE, 1000))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = net, labels = random_labels)
        optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, net, "Forward")

        # objective1 = tf.nn.l2_loss(end_points['AuxLogits'])
        # grad1 = tf.gradients(objective1)
        # objective2 = tf.nn.l2_loss(logits)
        # grad2 = tf.gradients(objective2)

        time_tensorflow_run(sess, optimizer, "Backward")

run_benchmark()
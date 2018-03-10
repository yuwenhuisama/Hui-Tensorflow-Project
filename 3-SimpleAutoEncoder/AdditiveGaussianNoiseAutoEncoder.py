'''
File: \AdditiveGaussianNoiseAutoEncoder.py
Project: 3- SimpleAutoEncoder
Created Date: Saturday March 10th 2018
Author: Huisama
-----
Last Modified: Saturday March 10th 2018 2:56:02 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import numpy as np
import tensorflow as tf

'''
    Xavier initializer
'''
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

'''
    AdditiveGaussianNoiseAutoEncoder Class

    Structure: Input Layer -> Hidden Layer -> Reconstruct Layer
'''
class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdadeltaOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        # Gauss noise coefficient
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        self.weights = self._initialize_weights()

        # define nn/loss function and start
        self._define_NN_structure()
        self._define_loss_functon(optimizer)
        self._start_session()

    def _define_NN_structure(self):
        # input layer
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # hidden layer
        self.hidden = self.transfer(
            tf.add(tf.matmul(
                # Give Gaussian Noise to the input
                self.x + self.scale * tf.random_normal((self.n_input,)),
                self.weights['w1']),
                self.weights['b1']
            )
        )

        self.hidden2 = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        
        # reconstruct layer
        self.reconstruction = tf.add(tf.matmul(self.hidden2, self.weights['w3']), self.weights['b3'])

    def _define_loss_functon(self, optimizer):
        # 1/2 * Sum of (x'-x)^2
        self.cost = 0.5 * tf.sqrt(tf.reduce_sum(
            tf.pow(
                tf.subtract(self.reconstruction, self.x), 2.0)
            ))
        self.optimizer = optimizer.minimize(self.cost)

    def _start_session(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = {}
        
        # Hidden layer (n_input inputs n_hidden outputs)
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        all_weights['w2'] = tf.Variable(xavier_init(self.n_hidden, self.n_hidden))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        # Reconstruct layer (n_hidden inputs n_input outputs)
        all_weights['w3'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b3'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        return all_weights

    '''
        partial_fit calc the cost and do optimization (from begin to end)
    '''
    def partial_fit(self, X):
        cost, _ = self.sess.run((self.cost, self.optimizer),
            feed_dict = {
                self.x: X,
                self.scale: self.training_scale,
            })
        return cost
    
    '''
        calc_total_cost just calc the cost (for testing)
    '''
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {
            self.x: X,
            self.scale: self.training_scale,
        })
    
    '''
        transform do calculation before hidden layer (from begin to hidden layer)
    '''
    def transform(self, X):
        return self.sess.run(self.hidden,
            feed_dict = {
                self.x: X,
                self.scale: self.training_scale
            }
        )

    '''
       generate do calculation before reconstruction layer (from hidden layer to reconstruction layer)
    '''
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction,
            feed_dict = { self.hidden: hidden }
        )

    '''
        reconstruct from begin to end
    '''
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
            feed_dict = {
                self.x: X,
                self.scale: self.training_scale
            }
        )

    def get_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_biases(self):
        return self.sess.run(self.weights['b1'])
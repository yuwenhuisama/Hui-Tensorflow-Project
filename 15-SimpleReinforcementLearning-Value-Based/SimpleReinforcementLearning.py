'''
File: \SimpleReinforcementLearning.py
Project: 15-SimpleReinforcementLearning-Value-Based
Created Date: Wednesday March 28th 2018
Author: Huisama
-----
Last Modified: Thursday March 29th 2018 11:12:49 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import tensorflow as tf
import numpy as np
import random
import os
import Game

env = Game.GameEnv(size = 5)

class Qnetwork():
    def __init__(self, h_size):
        self.scalarInput = tf.placeholder(shape = [None, 21168], dtype = tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape = [-1, 84, 84, 3])
        # output: 20x20x32
        self.conv1 = tf.contrib.layers.convolution2d(
            inputs = self.imageIn,
            num_outputs = 32,
            kernel_size = [8, 8],
            stride = [4, 4],
            padding = 'VALID',
            biases_initializer = None
        )
        # output: 9x9x64
        self.conv2 = tf.contrib.layers.convolution2d(
            inputs = self.conv1,
            num_outputs = 64,
            kernel_size = [4, 4],
            stride = [2, 2],
            padding = 'VALID',
            biases_initializer = None
        )
        # output: 7x7x64
        self.conv3 = tf.contrib.layers.convolution2d(
            inputs = self.conv2,
            num_outputs = 64,
            kernel_size = [3, 3],
            stride = [1, 1],
            padding = 'VALID',
            biases_initializer = None
        )
        # output: 1x1x512
        self.conv4 = tf.contrib.layers.convolution2d(
            inputs = self.conv3,
            num_outputs = 512,
            kernel_size = [7, 7],
            stride = [1, 1],
            padding = 'VALID',
            biases_initializer = None
        )

        # split the output of conv4 into streamAC and streamVC
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        # flatten them
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        # output for n actions
        self.AW = tf.Variable(tf.random_normal([h_size // 2, env.actions]))
        # output for value estimation
        self.VW = tf.Variable(tf.random_normal([h_size //2, 1]))

        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # qout = value + (advantage - mean of all advantages) batch x 4 (adv1, adv2, adv3, adv4)
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices = 1, keep_dims = True))
        # predict = index maxium of qout batch x 1
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype = tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices = 1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class ExperienceBuffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0: (len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

def processState(states):
    return np.reshape(states, [21168])

def updateTragetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    # 0...total_vars // 2 is parameters of main DQN
    # total_vars // 2 + 1 ... total_vars is parameters of target DQN
    # let parameters of target DQN to learn main DQN as:
    # TQ = a * TQ + (1 - a) * MQ
    for idx, var in enumerate(tfVars[0: total_vars // 2]):
        op_holder.append(
            tfVars[idx + total_vars // 2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value()))
        )
    return op_holder

def updateTraget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

# do tarining
# the number of samples which nn get from experience
batch_szie = 32
# the number of steps after witch nn update model parameters
update_freq = 4
# the discount factor of q value
y = 0.99
# the probability of randomly picking action when starting
startE = 1
# the probability of randomly picking action when ending
endE = 0.1
# steps from startE to endE
anneling_steps = 10000.
# the number of testing times under Game
num_episodes = 10000
# the number of steps before letting DQN select actions
pre_train_steps = 10000
# the number of steps of each episode
max_epLength = 50
# whether loading saved model
load_model = False
# the path of saved model
path = "./dqn"
# the number of hidden nodes
h_size = 512
# learning rate
tau = 0.001

mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)
init = tf.global_variables_initializer()

trainables = tf.global_variables()
targetOps = updateTragetGraph(trainables, tau)

myBuffer = ExperienceBuffer()
e = startE
stepDrop = (startE - endE) / anneling_steps

rList = []
total_steps = 0

saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        if (ckpt):
            saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(init)

    updateTraget(targetOps, sess)

    for i in range(num_episodes + 1):
        episodeBuffer = ExperienceBuffer()
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        while j < max_epLength:
            j += 1
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            else:
                a = sess.run(mainQN.predict, feed_dict = {
                    mainQN.scalarInput: [s]
                })[0]

            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                if total_steps % update_freq == 0:
                    trainBatch = myBuffer.sample(batch_szie)
                    A = sess.run(mainQN.predict, feed_dict = {
                        mainQN.scalarInput: np.vstack(trainBatch[:, 3])
                    })
                    Q = sess.run(targetQN.Qout, feed_dict = {
                        targetQN.scalarInput: np.vstack(trainBatch[:, 3])
                    })
                    doubleQ = Q[range(batch_szie), A]
                    targetQ = trainBatch[:, 2] + y * doubleQ
                    _ = sess.run(mainQN.updateModel, feed_dict = {
                        mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                        mainQN.targetQ: targetQ,
                        mainQN.actions: trainBatch[:, 1]
                    })

                    updateTraget(targetOps, sess)
            rAll += r
            s = s1

            if d == True:
                break

        myBuffer.add(episodeBuffer.buffer)
        rList.append(rAll)

        if i > 0 and i % 25 == 0:
            print('episode', i, ', average reward of last 25 episode', np.mean(rList[-25]))

        if i > 0 and i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print('Saved Model')

    saver.save(sess, path + '/model-' + str(i) + '.ctpk')

rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
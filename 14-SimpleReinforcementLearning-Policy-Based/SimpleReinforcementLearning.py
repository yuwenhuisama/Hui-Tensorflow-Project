'''
File: \SimpleReinforcementLearning.py
Project: 14-SimpleReinforcementLearning
Created Date: Friday March 23rd 2018
Author: Huisama
-----
Last Modified: Saturday March 24th 2018 9:21:56 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import numpy as np
import tensorflow as tf
import gym

env = gym.make('CartPole-v0')

# env.reset()
# random_episodes = 0
# reward_sum = 0
# while random_episodes < 10:
#     env.render()
#     observation, reward, done, _ = env.step(np.random.randint(0, 2))
#     reward_sum += reward

#     if done:
#         random_episodes += 1
#         print('Reward for this episode was: ', reward_sum)
#         reward_sum = 0
#         env.reset()

# env.close()

H = 50
batch_size = 25
learning_rate = 1e-1
D = 4
gamma = 0.99

# Stand for environment parameters
observations = tf.placeholder(tf.float32, [None, D], name = 'input_x')
W1 = tf.get_variable('W1', shape = [D, H], initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))

W2 = tf.get_variable('W2', shape = [H, 1], initializer = tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)

# Stand for the action will be done (< 0.5 = left, >= 0.5 = right)
probability = tf.nn.sigmoid(score)

def discount_rewards(r):
    discounted_r = np.zeros_like(r)

    running_add = 0
    # Earlier action is more valuable
    # For rewards list [r1, r2, .., rn]
    # Calculate its discount as:
    # d1 = rn
    # d2 = rn-1 + d1 * gamma = rn-1 + rn * gamma
    # d3 = rn-2 + d2 * gamma = rn-2 + (rn-1 + d1 * gamma) * gamma = rn-2 + rn-1 * gamma + rn * gamma ^ 2
    # ...
    # dn = r1 + r2 * gamma + r3 * gamma^2 + ... + rn * gamma^(n-1)
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# loglik = log(label * (label - prob) + ((1 - label) * (label + prob)))
# label = 0 or 1. when action = 0 lable = 1 while action = 1 label = 0
# prob stands for the probablity of action = 1
# 1 - prob stands for the probability of action = 0
# when action = 1, label = 0, loglik = log(prob) standing for the log probability of action = 1
# when action = 0, label = 1, loglik = log(1 - prob) standing for the log probablity of action = 0
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))

# note that loglik is nagative, so -loglik is positive
# when the probability of an action get larger, -loglik get smaller, so let loss = -loglik * advantage and minimize it, which tends to get more advantages
loss = -tf.reduce_mean(loglik * advantages)

tvars = tf.trainable_variables()
newGrads = tf.gradients(loss, tvars)

# optimizer
adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
W1Grad = tf.placeholder(tf.float32, name = 'batch_grade1')
W2Grad = tf.placeholder(tf.float32, name = 'batch_grade2')
batchGrade = [W1Grad, W2Grad]

# Update W1, W2
updateGrads = adam.apply_gradients(zip(batchGrade, tvars))

# Parameters for training
xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 10000

with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)

    observation = env.reset()

    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True

        x = np.reshape(observation, [1, D])

        # only predict the action with current parameters without training
        tfprob = sess.run(probability, feed_dict = { observations: x })

        # output of NN is a probablity value which means the probablity of taking action = 1
        # random a number in [0, 1] and check if it is in [0, probablity] (taking action = 1) or [probablity, 1] (taking action = 0)
        action = 1 if np.random.uniform() < tfprob else 0

        # Observation
        xs.append(x)

        # Label
        y = 1 - action
        ys.append(y)

        observation, reward, done, info = env.step(action)
        reward_sum += reward

        drs.append(reward)

        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)

            xs, ys, drs = [], [], []

            # (de - avg(de)) / std(de) ~ N(0, 1 )
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            tGrad = sess.run(newGrads, feed_dict = { observations: epx, input_y: epy, advantages: discounted_epr })

            # save all the grad of d(loss)/d(all vars in NN)
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict = {
                    W1Grad: gradBuffer[0],
                    W2Grad: gradBuffer[1]
                })

                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                print('Avarage reward for episode %d : %f.' % (episode_number, reward_sum / batch_size))

                # if avarage reward is larger than 200 then stop training.
                if reward_sum / batch_size > 200:
                    print('Task solved in', episode_number, ' episodes!')
                    break
                
                reward_sum = 0
            observation = env.reset()

env.close()
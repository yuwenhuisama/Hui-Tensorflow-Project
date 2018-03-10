'''
File: \test.py
Project: 1-Test
Created Date: Friday March 9th 2018
Author: Huisama
-----
Last Modified: Friday March 9th 2018 1:04:42 pm
Modified By: Huisama
-----
Copyright (c) 2018 Hui
'''

import tensorflow as tf

a = tf.random_normal((100, 100))
b = tf.random_normal((100, 500))
c = tf.matmul(a, b)
sess = tf.InteractiveSession()
sess.run(c)

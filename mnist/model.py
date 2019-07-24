# -*- coding: utf-8 -*-
# author：yaoyao time:2019/7/24

import tensorflow as tf

"""定义一个简单的线性模型"""
# Y = W*x+b
def regression(x):
    W = tf.Variable(tf.zeros([784,10]),name="w")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x,W)+b)

    return y,[W,b]

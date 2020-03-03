# -*- coding: utf-8 -*-

import tensorflow as tf


def layer (inputs, insize, outsize, activation = False):
    
    weights = tf.Variable(tf.random_normal([insize,outsize])) #权重值设定，随机且符合正态分布，
    #insize代表 input的一个张量 outsize代表输出的张量，也是神经元个数
    biases = tf.Variable(tf.zeros([1,outsize]) +0.1)
    #偏置的张量为一行但有很多列, 0.1 为noise
    
    wx_plus_b = tf.matmul(tf.cast(inputs,tf.float32),weights) + biases #将input转为float32
    
    if activation is False:
        outputs = wx_plus_b
        
    else:
        outputs = activation(wx_plus_b)
    return outputs
    
# -*- coding: utf-8 -*-
import numpy as np
import TF5_hidden_layer as tf5
import tensorflow as tf

x_data = np.linspace(-1.0,1.0,300)[:,np.newaxis] #np.newaxis是增加一个维度，shape为（300，1）

noise = np.random.normal(0,0.05,x_data.shape) #noise使用与x相同的shape
y_data = np.square(x_data)-0.5 + noise #目标
#print(y_data.shape,x_data.shape)

xs = tf.placeholder(tf.float32,[None,1])#设立传入置的变量, None 代表数量多少都可以
ys = tf.placeholder(tf.float32,[None,1])

layer_1 = tf5.layer(xs,1,10, activation = tf.nn.relu)
prediction = tf5.layer(layer_1, 10, 1, activation = False)

loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(ys - prediction),reduction_indices = [1]))
    ##求标准差


train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init =  tf.initialize_all_variables()
sess = tf.Session()
sess.run(init) #激活变量

for i in range(1000):
    sess.run(train_step,feed_dict ={xs:x_data,ys:y_data})
    
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs:x_data,ys:y_data}))
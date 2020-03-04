# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def layer (inputs, insize, outsize, n_layer, activation = False):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([insize,outsize]), name ='w') 
            tf.summary.histogram(layer_name+'/weights', weights)
            
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,outsize]) +0.1)
            tf.summary.histogram(layer_name+'/biases', biases)
        
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(tf.cast(inputs,tf.float32),weights) + biases
           
            
        if activation is False:
            outputs = wx_plus_b
        
        else:
            outputs = activation(wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs', biases)
        return outputs

    
x_data = np.linspace(-1.0,1.0,300)[:,np.newaxis] 

noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5 + noise 


with tf.name_scope('inputs'):
    
    xs = tf.placeholder(tf.float32,[None,1], name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1], name = 'y_input')

layer_1 = layer(xs,1,10, n_layer =1, activation = tf.nn.relu)
prediction = layer(layer_1, 10, 1,n_layer =2, activation = False)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(ys - prediction),reduction_indices = [1]))
    tf.summary.scalar('loss',loss)


train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init =  tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.compat.v1.summary.FileWriter('logs/',sess.graph)
sess.run(init) 

for i in range(1000):
    sess.run(train_step,feed_dict ={xs:x_data,ys:y_data})
    
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i) #i为步数，每50步记录一个点
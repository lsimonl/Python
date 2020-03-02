# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.2 + 0.5  # training target

Weights = tf.Variable(tf.random_uniform([1],-2,2)) #权重值必须随机，这里生成一个一维，从-2到+2的值
biases = tf.Variable(tf.zeros([1])) #偏置可以为常数 设为一维的0

y = Weights * x_data + biases   #我们的目标是将y训练成 与y_data接近

loss = tf.reduce_mean(tf.square(y-y_data)) #设置损失函数，对比y与ydata的区别，要减小这个函数的值
opt = tf.train.GradientDescentOptimizer(0.5) #设置变量= 梯度下降为训练方法.
train = opt.minimize (loss) #减小损失函数，设置训练步骤

init = tf.initialize_all_variables() #激活所有变量
sess = tf.Session() 
sess.run(init) #执行init， 激活变量

fig=plt.figure()               
plt.ion()
plt.show()

for i in range(181):
    sess.run(train) #执行训练
    if i % 20 == 0: #每20次训练执行：
        print(i,sess.run(Weights),sess.run(biases))
        plt.scatter(sess.run(Weights),sess.run(biases)) #画出权重值与偏置的动图
        plt.pause(0.3)
       



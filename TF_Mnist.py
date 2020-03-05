# -*- coding: utf-8 -*-
import tensorflow as tf
import TF5_hidden_layer as TF5
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)



#占位符设置 为训练，测试的所有input值
xs = tf.placeholder(tf.float32,[None,784])#784 =28*28 横竖像素相乘
ys = tf.placeholder(tf.float32,[None,10])#one hot 标记值的维度为10


#add output layer
prediction = TF5.layer(xs,784,10,tf.nn.softmax) #softmax一般做分类
#进入值为 [none,784] 分类结果为[none,10]

#分类的损失函数一般用交叉熵
cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(ys * tf.log(prediction),reduction_indices =[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#随机梯度下降SGD，后面用next_batch每次提取100个

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

def compute_accuracy(x_test,y_test): #评价模型准确率函数，输入为测试集
    global prediction  #神经层 prediction设置为全局变量
    
    y_prediction = sess.run(prediction,feed_dict={xs : x_test}) 
    #对y测试集的预测，要使用之前训练的模型prediction，输入要为x的测试集
    correction_prediction = tf.equal(tf.argmax(y_prediction,1),tf.argmax(y_test,1))
    # 对比预测的正确度， 需要对比测试集的预测值 以及 测试集的实际label值
    accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))
    #转换为概率的平均数
    result =sess.run(accuracy, feed_dict = {xs:x_test, ys:y_test})
    return result

for i in range(1000):
    batch_xs, batch_ys =mnist.train.next_batch(100) #每次随机抽取100个数 是SGD算法
    sess.run(train_step, feed_dict = {xs:batch_xs, ys:batch_ys})
    
    if i %50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
    
    

    
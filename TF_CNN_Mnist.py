# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def compute_accuracy(x_test,y_test): #评价模型准确率函数，输入为测试集
    global prediction  
    y_prediction = sess.run(prediction,feed_dict={xs : x_test})  #对y测试集的预测，要使用之前训练的模型prediction，输入要为x的测试集
    correction_prediction = tf.equal(tf.argmax(y_prediction,1),tf.argmax(y_test,1)) # 对比预测的正确度， 需要对比测试集的预测值 以及 测试集的实际label值
    accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32)) #转换为概率的平均数
    result =sess.run(accuracy, feed_dict = {xs:x_test, ys:y_test})
    return result

def weight_variable(shape): #所有层的权重设置
    init = tf.truncated_normal(shape, stddev = 0.1) #有切割的正态分布随机数，标准差为0.1
    return tf.Variable(init)

def bias_variable(shape): #所有层的偏置设置
    init = tf.constant(0.1, shape = shape)  #bias可以为常数，通常为正值比较好
    return tf.Variable(init)
    
def conv_layer_2d(x,w):
    # 步长 strides = [1, x_movement, y_movement, 1] 首尾都必须为1
    return tf.nn.conv2d(x,w,strides = [1,1,1,1],padding ='SAME') #padding：same与原图片大小一致
    
def max_pool_2x2(x):
    #ksize是池化盒的size， 通常为2，且与步长一致
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding='SAME')


xs = tf.placeholder(tf.float32,[None,784]) #print(xs[0:1].shape) =28*28 横竖像素相乘
x_image = tf.reshape(xs,[-1,28,28,1]) #维度转换: -1代表忽略数量，28代表width，height，1代表单色通道
ys = tf.placeholder(tf.float32,[None,10])#one hot 标记值的维度为10


###################第一层卷积与池化############################
w_conv1 = weight_variable([5,5,1,32]) 
#设置权重值,5是patch 卷积核size，1是单通道输入，32是32通道输出（输出值）
b_conv1 = bias_variable([32]) #bias = output_size =卷积核数，设置偏置。
hidden_conv1 = tf.nn.relu(conv_layer_2d(x_image, w_conv1) + b_conv1)
#卷积层output = 28*28*32 Padding使用SAME, 原图像大小不变
hidden_pool1 = max_pool_2x2(hidden_conv1)
#池化output = 14*14*32 因为步长是卷积的两倍，所以图片size变为1/2

####################第二层卷积与池化#########################################
w_conv2 = weight_variable([5,5,32,64]) #传入为32来自于上一层，假设传出为64，把他变高
b_conv2 = bias_variable([64])
hidden_conv2 = tf.nn.relu(conv_layer_2d(hidden_pool1 ,w_conv2)+ b_conv2) 
#卷积output = 14*14*64
hidden_pool2 = max_pool_2x2(hidden_conv2)
#池化output = 7*7*64

#################全连接层1 FC(fully connected) 全连接层需要展开张量#############
hidden_pool2_flat = tf.reshape(hidden_pool2,[-1,7*7*64])
#-1代表数量无所谓 由[n,7,7,64] 变为[n,7*7*64]
w_FC1 = weight_variable([7*7*64,1024]) #假定输出size为1024
b_FC1 = bias_variable([1024])
hidden_FC1 = tf.nn.relu(tf.matmul(hidden_pool2_flat,w_FC1)+b_FC1)

####################全连接层2#################################################
w_FC2 = weight_variable([1024,10]) #由上层传导
b_FC2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(hidden_FC1,w_FC2) + b_FC2)


#分类损失函数一般用交叉熵
cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(ys * tf.log(prediction),reduction_indices =[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)



for i in range(1000):
    batch_xs, batch_ys =mnist.train.next_batch(100) #每次随机抽取100个数 
    sess.run(train_step, feed_dict = {xs:batch_xs, ys:batch_ys})
    
    if i %50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))
    
    

    
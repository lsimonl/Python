# -*- coding: utf-8 -*-


def compute_accuracy(x_test,y_test): #评价模型准确率函数，输入为测试集
    global prediction  #神经层 prediction设置为全局变量
    
    y_prediction = sess.run(prediction,feed_dict={xs : x_test}) 
    #对y测试集的预测，要使用之前训练的神经层prediction，输入要为x的测试集
    correction_prediction = tf.equal(tf.argmax(y_prediction,1),tf.argmax(y_test,1))
    # 对比预测的正确度， 需要对比测试集的预测值 以及 测试集的实际label值
    accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))
    #转换为概率的平均数
    result =sess.run(accuracy, feed_dict = {xs:x_test, ys:y_test})
    return result
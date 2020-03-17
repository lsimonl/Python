# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score, precision_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
#print(iris)
#查看数据集 字典分为data 和 target，data具有4个属性，target具有三个类别
# 因此要根据data的属性，判断属于鸢尾花的哪个类别

x_data = iris['data']
y_data = iris['target']

def cross_validation():
    x_train, x_test, y_train, y_test = train_test_split(
            x_data,y_data,test_size=0.3)
    return x_train,x_test,y_train,y_test

cross_validation()
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)  #fit 类方法用来训练模型
prediction = knn.predict(x_test) #predict 用来预测测试集
#print(prediction,y_test)

def accuracy(y_test_value,prediction_value):
    accuracy_prob = accuracy_score(y_test_value,prediction_value)
    print(accuracy_prob)
    
accuracy(y_test,prediction)

#print(precision_score(y_test,prediction,average ='macro'))  另一种accuracy的measurement
    
    
    
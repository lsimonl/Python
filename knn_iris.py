# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import pickle
#######################数据加载及标准化#######################
iris = datasets.load_iris()
#查看数据集 字典分为data 和 target，data具有4个属性，target具有三个类别
#因此要根据data的属性，判断属于鸢尾花的哪个类别,属于分类算法
x_data = iris['data']
x_data = preprocessing.scale(x_data)
y_data = iris['target']

######################训练集测试集分开##########################
x_train, x_test, y_train, y_test = train_test_split(
        x_data,y_data,test_size=0.3)


########################模型调参####################
estimator = KNeighborsClassifier()  #调用哪一类算法
param_dict = {'n_neighbors':range(1,31)}  #设立超参数词典
estimator = GridSearchCV(estimator,param_grid = param_dict,cv=5)
#网格搜索调参，5折交叉验证,并且传入到新的estimator对象中


#########################训练模型###################
estimator.fit(x_train,y_train)  #fit类方法用来训练模型

#####################评估模型在测试集################################
prediction = estimator.predict(x_test) #predict 用来预测测试集
def accuracy(y_test_value,prediction_value):
   accuracy_prob = accuracy_score(y_test_value,prediction_value)
   print(accuracy_prob)

accuracy(y_test,prediction)
print(estimator.score(x_test,y_test))
#print(precision_score(y_test,prediction,average ='macro'))  其他两种accuracy的measurement
#print(prediction,y_test)    

#########################训练模型保存#####################
with open('save/iris_model.pickle','wb') as f:
    pickle.dump(estimator,f)
    
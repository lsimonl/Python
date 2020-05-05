# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

##############预处理################
df = pd.read_csv("https://sololearn.com/uploads/files/titanic.csv")
#print(df.head())
df['male'] = df['Sex'] =='male' #建立新的列，将sex转为boolean值
#print(df.head())

x = df[['Pclass','Age','Siblings/Spouses','Parents/Children','Fare','male']].values
y = df['Survived'].values                 #pandas series要转化为 numpy array

############训练模型#############
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)
model = LogisticRegression()
model.fit(x_train,y_train)
#print(model.coef_, model.intercept_)
y_prediction1 = model.predict(x_train)
print(model.score(x_train,y_train)) #训练集准确率

#############预测#############################
y_prediction2 = model.predict(x_test)
#print((y_prediction2 == y_test).sum() / y_test.shape[0])  #测试集准确率
print(model.score(x_test, y_test))

print(model.predict([[3,26,0,0,7.925,False]]))  #对某个人的predict

############保存模型###########################
with open('save/logisticRegression.pickle','wb') as f:
    pickle.dump(model,f)
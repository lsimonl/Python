# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split


def cross_validation(x_dataValue, y_dataValue):
    x_train, x_test, y_train, y_test = train_test_split(
            x_dataValue,y_dataValue,test_size=0.3)
    return x_train,x_test,y_train,y_test

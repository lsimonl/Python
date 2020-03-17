# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split


def cross_validation(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(
            x_data,y_data,test_size=0.3)
    return x_train,x_test,y_train,y_test

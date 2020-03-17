# -*- coding: utf-8 -*-
def accuracy(y_test_value,prediction_value):
    accuracy_prob = accuracy_score(y_test_value,prediction_value)
    print(accuracy_prob)
#coding:utf-8
import numpy as np

train_X = np.array([[3,3],[4,3],[1,1]])
train_Y = np.array([1,1,-1])

W = np.zeros(2);
b = np.zeros(1);

for j in range(7):
    for i in range(3):
        Y = train_Y[i] * (np.dot(W,train_X[i]) + b)
        if Y <= 0:
            temp = train_Y[i] * train_X[i]
            W = W + temp
            b = b + train_Y[i]
        print W
        print b    
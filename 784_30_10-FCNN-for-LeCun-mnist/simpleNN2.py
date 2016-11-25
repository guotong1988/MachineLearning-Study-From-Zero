# coding:utf-8
import numpy as np
import mnist_loader
# 用full-connected神经网络搞的LeCun的手写数字识别

# 784*30*10

speed = 1

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1.0 - sigmoid(z))

W = []
W1 = np.random.randn(30, 784)
W2 = np.random.randn(10, 30)
W.append(W1)
W.append(W2)

b = []
b1 = np.random.randn(30,1)
b2 = np.random.randn(10,1)
b.append(b1)
b.append(b2)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
 
for i in range(0, 50000):
    zs = []
    activations = []
    activations.append(training_data[i][0])
    z0 = np.dot(W[0], training_data[i][0]) + b[0]
    zs.append(z0)
    a1 = sigmoid(z0)
    activations.append(a1)
    z1 = np.dot(W[1], a1) + b[1]
    zs.append(z1)
    a2 = sigmoid(z1)
    activations.append(a2)
    
    error = []
    
    error0 = activations[2] - training_data[i][1]
    error0 = error0 * sigmoid_prime(zs[1])
    error.append(error0)

    error1 = np.dot(W[1].transpose(),error[0])
    error1 = error1 * sigmoid_prime(zs[0])
    error.append(error1)
    
    print i
    
    dw0 = np.dot(error[1], activations[0].transpose())

    dw1 = np.dot(error[0], activations[1].transpose())
 
 
    W[0] = W[0] - dw0    
    b[0] = error[1] - b[0]
    W[1] = W[1] - dw1
    b[1] = error[0] - b[1]
    
  
            


count = 0
for i in range(100):
    a = test_data[i][0]
    for bb, ww in zip(b, W):
        a = sigmoid(np.dot(ww, a) + bb)    
    print a
    print test_data[i][1]
    if np.argmax(a) == test_data[i][1]:
        count = count + 1
print count

#coding:utf-8
import numpy as np
#https://en.wikipedia.org/wiki/Backpropagation
#http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

#4*3*2

speed = 1

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

w1 = []
w1.append([])
w1[0].append(0.1)
w1[0].append(0.2)
w1[0].append(0.3)
w1.append([])
w1[1].append(0.11)
w1[1].append(0.22)
w1[1].append(0.33)
w1.append([])
w1[2].append(0.5)
w1[2].append(0.6)
w1[2].append(0.7)
w1.append([])
w1[3].append(0.55)
w1[3].append(0.66)
w1[3].append(0.77)

w2 = []
w2.append([])
w2[0].append(0.1)
w2[0].append(0.2)
w2.append([])
w2[1].append(0.3)
w2[1].append(0.4)
w2.append([])
w2[2].append(0.5)
w2[2].append(0.6)

W1 = np.array([ 
      [w1[0][0],w1[0][1],w1[0][2]],
      [w1[1][0],w1[1][1],w1[1][2]],
      [w1[2][0],w1[2][1],w1[2][2]],
      [w1[3][0],w1[3][1],w1[3][2]] 
      ])

W2 = np.array([
      [w2[0][0],w2[0][1]],
      [w2[1][0],w2[1][1]],
      [w2[2][0],w2[2][1]],
      ])

X = []
Y = []

X.append(np.array([1,1,1,1]))
Y.append([1,1])

X.append(np.array([-1,-1,-1,-1]))
Y.append([0,0])

X.append(np.array([1,1,-1,-1]))
Y.append([1,0])

X.append(np.array([-1,-1,1,1]))
Y.append([0,1])

for i in range(1,10000):
    zs = []
    activations = []
    print X[i%4]
    z1 = np.dot(W1.transpose(),X[i%4])
    zs.append(z1)
    a1 = sigmoid(z1)
    activations.append(a1)
    z2 = np.dot(W2.transpose(),a1)
    zs.append(z2)
    a2 = sigmoid(z2)
    activations.append(a2)
    print a2    
    error = []
    error.append([])    
    error[0].append(Y[i%4][0] - a2[0])
    error[0].append(Y[i%4][1] - a2[1])
    error.append([])
        
    error[1].append(error[0][0]*W2[0][0]+error[0][1]*W2[0][1])
    error[1].append(error[0][0]*W2[1][0]+error[0][1]*W2[1][1])
    error[1].append(error[0][0]*W2[2][0]+error[0][1]*W2[2][1])
    
    W1[0][0] = sigmoid_prime(zs[0][0])*X[i%4][0]*error[1][0]*speed + W1[0][0]
    W1[0][1] = sigmoid_prime(zs[0][1])*X[i%4][0]*error[1][1]*speed + W1[0][1]
    W1[0][2] = sigmoid_prime(zs[0][2])*X[i%4][0]*error[1][2]*speed + W1[0][2]
    
    W1[1][0] = sigmoid_prime(zs[0][0])*X[i%4][1]*error[1][0]*speed + W1[1][0]
    W1[1][1] = sigmoid_prime(zs[0][1])*X[i%4][1]*error[1][1]*speed + W1[1][1]
    W1[1][2] = sigmoid_prime(zs[0][2])*X[i%4][1]*error[1][2]*speed + W1[1][2]
    
    W1[2][0] = sigmoid_prime(zs[0][0])*X[i%4][2]*error[1][0]*speed + W1[2][0]
    W1[2][1] = sigmoid_prime(zs[0][1])*X[i%4][2]*error[1][1]*speed + W1[2][1]
    W1[2][2] = sigmoid_prime(zs[0][2])*X[i%4][2]*error[1][2]*speed + W1[2][2]
    
    W1[3][0] = sigmoid_prime(zs[0][0])*X[i%4][3]*error[1][0]*speed + W1[3][0]
    W1[3][1] = sigmoid_prime(zs[0][1])*X[i%4][3]*error[1][1]*speed + W1[3][1]
    W1[3][2] = sigmoid_prime(zs[0][2])*X[i%4][3]*error[1][2]*speed + W1[3][2]
    
    W2[0][0] = sigmoid_prime(zs[1][0])*activations[0][0]*error[0][0]*speed + W2[0][0]
    W2[0][1] = sigmoid_prime(zs[1][1])*activations[0][0]*error[0][1]*speed + W2[0][1]
    
    W2[1][0] = sigmoid_prime(zs[1][0])*activations[0][1]*error[0][0]*speed + W1[1][0]
    W2[1][1] = sigmoid_prime(zs[1][1])*activations[0][1]*error[0][1]*speed + W1[1][1]
    
    W2[2][0] = sigmoid_prime(zs[1][0])*activations[0][2]*error[0][0]*speed + W1[2][0]
    W2[2][1] = sigmoid_prime(zs[1][1])*activations[0][2]*error[0][1]*speed + W1[2][1]
    

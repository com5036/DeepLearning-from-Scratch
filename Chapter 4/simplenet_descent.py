import sys, os

from common.gradient import numerical_gradient

sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = SimpleNet()

f = lambda w: net.loss(x, t)

for i in range(20):
    dW = numerical_gradient(f, net.W)
    print('net.W :\n ', net.W)
    print('softmax = ', softmax(net.predict(x)))
    print('loss = ', net.loss(x, t))
    net.W -= dW

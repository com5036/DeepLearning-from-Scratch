import matplotlib.pylab as plt
import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int32)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.show()

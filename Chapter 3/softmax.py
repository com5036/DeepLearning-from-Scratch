import numpy as np


def softmax(a):
    c = np.max(a)  # prevent overflow
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([-0.3, 1, 4.45])
y = softmax(a)

print("softmax(a) = ", y)
print("sum is always '1' :", sum(y))

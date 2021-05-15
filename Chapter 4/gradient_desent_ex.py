from functions import gradient_descent
import numpy as np
import matplotlib.pylab as plt


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])

x, x_history = gradient_descent(function_2, init_x, lr=0.1, step_num=50)

plt.plot([-5, 5], [0, 0], 'b')
plt.plot([0, 0], [-5, 5], 'b')
plt.plot((x_history[:, 0]), x_history[:, 1], 'o')

plt.show()

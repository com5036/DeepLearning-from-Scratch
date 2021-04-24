import numpy as np

A = np.array([[1, 2], [3, 4]])

B = np.array([10, 20])

X = A.flatten()
print(X)

print(X[X > 2])

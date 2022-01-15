import numpy as np
from main import A_hat_matrix

N = 2**8
h = 1/N
epsilon = 0.5

A = A_hat_matrix(N,h,epsilon)
D, X = np.linalg.eig(A)
X_inv = np.linalg.inv(X)
D = np.diag(D)

K = np.linalg.norm(X, ord=2)*np.linalg.norm(X_inv, 2)
print(f"K = {K}")
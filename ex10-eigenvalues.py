import numpy as np
from matplotlib import pyplot as plt
from main import *

def jacobi_eigenvalues(N, h, epsilon):
    A = A_matrix(N,h,epsilon)
    Dinv  = np.diag(1/np.diag(A))
    Bjac = np.identity(N+1) - np.matmul(Dinv, A)
    print(Bjac)
    w, v = np.linalg.eig(Bjac)
    return w


N=10
h=1/N
epsilon = 0.5
w = jacobi_eigenvalues(N, h, epsilon)

plt.scatter(np.arange(len(w)), np.sort(w)[::-1])

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from main import *

def jacobi_eigenvalues(N, h, epsilon):
    A = A_hat_matrix(N,h,epsilon)
    Dinv  = np.diag(1/np.diag(A))
    Bjac = np.identity(N-1) - np.matmul(Dinv, A)
    w, v = np.linalg.eig(Bjac)
    return w


def compute_spectral_radius(w):
    return max(abs(w))


print("epsilon", "N", "spectral radius", sep = '\t')
for epsilon in [1, 0.5, 0.1, 0.01, 0.001]:
    print(epsilon)
    for N in [2**n for n in range(3,10)]:
        h = 1/N
        Bjac = jacobi_eigenvalues(N, h, epsilon)
        r = compute_spectral_radius(Bjac)
        # print(epsilon, N, r, sep="\t")
        print(round(r, 5), end = '\t')

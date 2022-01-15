import numpy as np
import matplotlib.pyplot as plt
from main import *

def gs_eigenvalues(N, h, epsilon):
    A = A_hat_matrix(N,h,epsilon)
    DE = np.tril(A) 
    DEinv = np.linalg.inv(DE)
    Bgs = np.identity(N-1) - np.matmul(DEinv, A)
    w, v = np.linalg.eig(Bgs)
    return w


def compute_spectral_radius(w):
    return max(abs(w))


print("epsilon", "N", "spectral radius", sep = '\t')
for epsilon in [1, 0.5, 0.1, 0.01, 0.001]:
    print(epsilon)
    for N in [2**n for n in range(3,10)]:
        h = 1/N
        Bgs = gs_eigenvalues(N, h, epsilon)
        r = compute_spectral_radius(Bgs)
        # print(epsilon, N, r, sep="\t")
        print(round(r, 5), end = '\t')

import numpy as np
import matplotlib.pyplot as plt
from main import *

def gs_eigenvalues(N, h, epsilon):
    A = A_hat_matrix(N,h,epsilon)
    DF = np.triu(A) 
    DFinv = np.linalg.inv(DF)
    Bbgs = np.identity(N-1) - np.matmul(DFinv, A)
    w, v = np.linalg.eig(Bbgs)
    return w


def compute_spectral_radius(w):
    return max(abs(w))


print("epsilon", "N", "spectral radius", sep = '\t')
for epsilon in [1, 0.5, 0.1, 0.01, 0.001]:
    print(epsilon)
    for N in [2**n for n in range(3,10)]:
        h = 1/N
        Bbgs = gs_eigenvalues(N, h, epsilon)
        r = compute_spectral_radius(Bbgs)
        # print(epsilon, N, r, sep="\t")
        print(round(r, 5), end = '\t')

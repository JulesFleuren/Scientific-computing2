import numpy as np
import matplotlib.pyplot as plt
from main import *

def jacobi_eigenvalues(N, h, epsilon):
    A = A_matrix(N,h,epsilon)
    D = np.diag(A)
    Bjac = np.identity(N+1) - np.matmul(A, 1/D)
    w, v = np.linalg.eig(Bjac)
    return w


def compute_spectral_radius(w):
    return max(abs(w))


print("epsilon", "N", "spectral radius", sep = '\t')
for epsilon in [0.5, 0.1, 0.01]:
    for N in [2**n for n in range(3,10)]:
        h = 1/N
        Bjac = jacobi_eigenvalues(N, h, epsilon)
        r = compute_spectral_radius(Bjac)
        print(epsilon, N, r, sep="\t")
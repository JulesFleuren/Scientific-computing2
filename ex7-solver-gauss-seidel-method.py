import numpy as np

from matplotlib import pyplot as plt

from main import A_matrix, f_N

def single_step_Gauss_seidel(A,f,u):
    for i in range(len(f)):
        u[i] = u[i] + (f[i] - np.sum(A[i,:]*u))/A[i,i]
    return u

def Gauss_seidel_iteration_method(A, f, TOL):
    u = np.zeros(len(f))

    k = 0
    sr = []
    res = 2*TOL
    while res > TOL:
        # u = single_step_Gauss_seidel(A,f,u)
        for i in range(len(f)):
            u[i] = u[i] + (f[i] - np.sum(A[i,:]*u))/A[i,i]

        res = np.linalg.norm(f - np.matmul(A, u))/np.linalg.norm(f)
        sr.append(res)
        k += 1
    return u, sr, k
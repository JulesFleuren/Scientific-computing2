import numpy as np
import matplotlib.pyplot as plt

from time import time

import ex6_solver_jacobi_method_cython as jac
import ex7_solver_gauss_seidel_method_cython as gs
import ex8_solver_backward_gauss_seidel_method_cython as bgs
import ex9_solver_symmetric_gauss_seidel_method_cython as sgs
from timeit import timeit

def direct_solve(N,h,epsilon = 1):
    A = A_matrix(N,h,epsilon)

    f = np.zeros(N+1)
    f[0] = 1

    u = np.linalg.solve(A,f)
    x = np.linspace(0,1,N+1)
    return x, u

def A_matrix(N,h,epsilon):
    A = np.zeros((N+1,N+1))
    np.fill_diagonal(A, (2*epsilon + h)/h**2)
    A[0,0] = 1
    A[-1,-1] = 1
    A += np.diag(np.full(N,-epsilon/h**2),1)
    A[0,1] = 0

    A += np.diag(np.full(N,(-epsilon-h)/h**2),-1)
    A[-1,-2] = 0
    return A

def f_N(N):
    f = np.zeros(N+1)
    f[0] = 1
    return f

def u_ex(x, epsilon):
    return (np.exp(x/epsilon) - np.exp(1/epsilon))/(1 - np.exp(1/epsilon))

def plot_scaled_residual(A, f, N, TOL, iter_method):
    """
    Plots the scaled residual for the given iteration method.
    """
    # Solve the linear system.
    u, rs, k = iter_method(A, f, TOL)

    # Plot the scaled residual.
    plt.figure()
    plt.plot(range(k), rs)
    plt.title(f"Plot of the scaled residual for the {iter_method.__name__} for N  = {N}.")
    plt.xlabel('Iteration')
    plt.ylabel('Scaled Residual')
    plt.yscale('log')
    plt.show()

    # Calculate the last 5 reduction factors.
    redf = [rs[-i]/rs[-i-1] for i in range(1,6)]
    return redf

if __name__ == "__main__":
    N = 2**7
    h = 1/N
    epsilon = 0.1

    A = A_matrix(N, h, epsilon)
    f = f_N(N)
    TOL = 1e-6
    time0 = time()
    u1, _, k1 = jac.jacobi_iteration_method(A,f,TOL)
    time1 = time()
    u2, _, k2 = gs.gauss_seidel_iteration_method(A,f,TOL)
    time2 = time()
    u3, _, k3 = bgs.backward_gauss_seidel_iteration_method(A,f,TOL)
    time3 = time()
    u4, _, k4 = sgs.symmetric_gauss_seidel_iteration_method(A,f,TOL)
    time4 = time()

    x = np.linspace(0,1,N+1)
    u_ref = u_ex(x, epsilon)

    print(k1, k2)

    plt.plot(x, u1, label=f"Jacobi ({k1} iterations, {time1-time0:.2f}s)")
    plt.plot(x, u2, label=f"forward Gauss-Seidel ({k2} iterations, {time2-time1:.2f}s)")
    plt.plot(x, u3, label=f"backward Gauss-Seidel ({k3} iterations, {time3-time2:.2f}s)")
    plt.plot(x, u4, label=f"symmetric Gauss-Seidel ({k4} iterations, {time4-time3:.2f}s)")
    plt.plot(x, u_ref, label="reference solution")

    plt.legend()
    plt.show()
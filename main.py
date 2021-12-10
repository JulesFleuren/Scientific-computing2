import numpy as np
import matplotlib.pyplot as plt

import ex7_solver_gauss_seidel_method_cython as gs_cy
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
    pass
    # print(gs_cy.gauss_seidel_iteration_method(np.array([[1.0, 2.0],[3.0,4.0]]), np.array([5.0,6.0]), 7))
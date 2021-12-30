import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from time import time

import ex6_solver_jacobi_method_cython as jac
import ex7_solver_gauss_seidel_method_cython as gs
import ex8_solver_backward_gauss_seidel_method_cython as bgs
import ex9_solver_symmetric_gauss_seidel_method_cython as sgs
import ex12_solver_GMRES as gmres
import ex13_solver_repeated_GMRES as rgmres

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

def plot_scaled_residual(iter_method, *args, **kwargs):
    """
    Plots the scaled residual for the given iteration method.
    """
    # Solve the linear system.
    u, rs, k = iter_method(*args, **kwargs)

    # Plot the scaled residual.
    plt.figure()
    plt.plot(range(k), rs)
    plt.title(f"Plot of the scaled residual for the {iter_method.__name__} for N  = {len(u)-1}.")
    plt.xlabel('Iteration')
    plt.ylabel('Scaled Residual')
    plt.yscale('log')
    plt.show()

    # Calculate the last 5 reduction factors.
    redf = [rs[-i]/rs[-i-1] for i in range(1,6)]
    return redf

def plot_graph_and_scaled_residual(iter_method, x, ax_graph, ax_residual, *args, **kwargs):
    """
    Plots the scaled residual for the given iteration method.
    """
    # Solve the linear system.
    time0 = time()
    u, rs, k = iter_method(*args, **kwargs)
    time1 = time()

    ax_graph.plot(x, u, label=f"{iter_method.__name__} ({k} iterations, {time1-time0:.2f}s)")

    # Plot the scaled residual.
    ax_residual.plot(range(k), rs, label=f"{iter_method.__name__}")

    
if __name__ == "__main__":
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    
    N = 2**8
    h = 1/N
    epsilon = 0.1

    A = A_matrix(N, h, epsilon)
    f = f_N(N)

    u_0 = np.zeros(f.shape)
    TOL = 1e-6

    x = np.linspace(0,1,N+1)
    u_ref = u_ex(x, epsilon)

    ax1.plot(x, u_ref, label="reference solution")

    plot_graph_and_scaled_residual(jac.jacobi_iteration_method, x, ax1, ax2, A, f, TOL, tridiagonal=True)
    plot_graph_and_scaled_residual(gs.gauss_seidel_iteration_method, x, ax1, ax2, A, f, TOL, tridiagonal=True)
    plot_graph_and_scaled_residual(bgs.backward_gauss_seidel_iteration_method, x, ax1, ax2, A, f, TOL, tridiagonal=True)
    plot_graph_and_scaled_residual(sgs.symmetric_gauss_seidel_iteration_method, x, ax1, ax2, A, f, TOL, tridiagonal=True)
    plot_graph_and_scaled_residual(gmres.GMRES_method, x, ax1, ax3, A, f, u_0, TOL)
    plot_graph_and_scaled_residual(rgmres.repeated_GMRES_method, x, ax1, ax3, A,f,u_0,TOL, 10)
    
    ax1.set_title(f"plot of solutions for N={N}")
    ax1.set_xlabel('x')
    ax1.set_ylabel('u')
    ax1.legend()

    ax2.set_title(f"Plot of the scaled residual for N  = {N}.")
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Scaled Residual')
    ax2.set_yscale('log')
    ax2.legend()

    ax3.set_title(f"Plot of the scaled residual for N  = {N}.")
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Scaled Residual')
    ax3.set_yscale('log')
    ax3.legend()

    plt.show()
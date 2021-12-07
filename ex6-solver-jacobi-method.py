import numpy as np

from matplotlib import pyplot as plt

from main import A_matrix, f_N

def jacobi_iteration_method(A, f, TOL):
    u = np.zeros(len(f))

    k = 0
    sr = []
    res = 2*TOL
    while res > TOL:
        u = u + (f - np.matmul(A, u))/np.diag(A)

        res = np.linalg.norm(f - np.matmul(A, u))/np.linalg.norm(f)
        sr.append(res)
        k += 1
    return u, sr, k

def plot_scaled_residual(A, f, N, TOL):
    """
    Plots the scaled residual for the Jacobi method.
    """
    # Solve the linear system.
    u, rs, k = jacobi_iteration_method(A, f, TOL)

    # Plot the scaled residual.
    plt.figure()
    plt.plot(range(k), rs)
    plt.title(f"Plot of the scaled residual for the Jacobi method for N  = {N}.")
    plt.xlabel('Iteration')
    plt.ylabel('Scaled Residual')
    plt.yscale('log')
    plt.show()

    # Calculate the last 5 reduction factors.
    redf = [rs[-i]/rs[-i-1] for i in range(1,6)]
    return redf

TOL = 1e-6; epsilon = .1
Ns = [2**n for n in range(3, 7)]
red = np.zeros([len(Ns), 5])
for k in range(len(Ns)):
    N = Ns[k]
    h=1/N
    redk = plot_scaled_residual(A_matrix(N, h, epsilon), f_N(N), N, TOL)
    red[k] = redk

print(red)

# Plot into 1 image
# Vary epsilon
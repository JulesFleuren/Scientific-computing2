import numpy as np

from matplotlib import pyplot as plt

from main import A_matrix, f_N

# def jacobi_method(A, f, N, TOL):
#     """
#     Solves the linear system Au = f using the Jacobi method.
#     """
#     # Initialize the solution vector. Guess u0 to be the constant vector equal to zero.
#     u = [0] * (N + 1)
#     # Initialize the residual vector.
#     r = [0]
#     # Initialize the error.
#     error = np.linalg.norm(f)
#     # Initialize the iteration counter.
#     k = 0
#     # Iterate until the error is less than the tolerance.
#     while error > TOL:
#         # Update the iteration counter.
#         k += 1
#         for i in range(N):
#             # Update the solution vector.
#             u[i] = (f[i] - sum(A[i][j] * u[j] for j in range(N))) / A[i][i]
#             # Update the residual vector.
#         r.append(f - A*u)
#         # Update the error (Scaled residuals).
#         error = np.linalg.norm(r[k])/np.linalg.norm(f)
#         print(k, error, np.linalg.norm(r[k]), np.linalg.norm(f))
#         error = 10*TOL/k
#     # Return the solution vector, the residual vector and the number of iterations.
#     return u, r, k

# def jacobi_iteration_method(A, f, N, TOL): # A is the matrix, f is the vector, N is the number of iterations, TOL is the tolerance
#     x = np.linspace(0,1,N+1)
#     u = np.zeros(N+1)
#     # u[0] = 1
#     u_new = np.zeros(N+1)
#     # u_new[0] = 1
#     r = []
#     k = 0
#     error = np.linalg.norm(f)
#     while error > TOL:
#         u = u_new
#         for i in range(1,N):
#             u_new[i] = (f[i] - sum(A[i][j] * u[j] for j in range(N)) - A[i][i]*u[i]) / A[i][i]
#             # u_new[i] = (1/A[i,i])*(f[i] - A[i,i-1]*u[i-1] - A[i,i+1]*u[i+1])
#         r.append(f - A*u_new)
#         error = np.linalg.norm(r[k])/np.linalg.norm(f)
#         print(k, np.linalg.norm(r[k]), error)
#         k += 1
#         error = 10*TOL/k
#     return x, u_new

def jacobi_iteration_method(A, f, N, TOL):
    u = np.zeros(N+1)
    
    # error = np.linalg.norm(f)

    k = 0
    sr = []
    res = 2*TOL
    c = 0 # Number of iterations before convergence
    while res > TOL:
        z = u
        for i in range(0,N):
            z[i] = u[i] + (f[i] - sum(A[i][j] * u[j] for j in range(N))) / A[i][i]

        u = z
        res = np.linalg.norm(f - np.matmul(A, u))/np.linalg.norm(f)
        sr.append(res)
        k += 1
    return u, sr, k

def plot_scaled_residual(A, f, N, TOL):
    """
    Plots the scaled residual for the Jacobi method.
    """
    # Solve the linear system.
    u, rs, k = jacobi_iteration_method(A, f, N, TOL)

    # Plot the scaled residual.
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
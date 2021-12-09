import numpy as np

from matplotlib import pyplot as plt

from main import A_matrix, f_N, plot_scaled_residual

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


if __name__ == "__main__":
    TOL = 1e-6; epsilon = .1
    Ns = [2**n for n in range(3, 7)]
    red = np.zeros([len(Ns), 5])
    for k in range(len(Ns)):
        N = Ns[k]
        h=1/N
        redk = plot_scaled_residual(A_matrix(N, h, epsilon), f_N(N), N, TOL, jacobi_iteration_method)
        red[k] = redk

    print(red)

# Plot into 1 image
# Vary epsilon
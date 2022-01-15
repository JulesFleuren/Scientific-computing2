import numpy as np

from matplotlib import pyplot as plt

from main import A_hat_matrix, f_hat_N, plot_scaled_residual


def single_step_gauss_seidel(A,f,u):
    for i in range(len(f)):
        u[i] = u[i] + (f[i] - np.sum(A[i,:]*u))/A[i,i]
    return u

def gauss_seidel_iteration_method(A, f, TOL):
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

if __name__ == "__main__":
    TOL = 1e-6; epsilon = .1
    Ns = [2**n for n in range(3, 7)]
    red = np.zeros([len(Ns), 5])
    for k in range(len(Ns)):
        N = Ns[k]
        h=1/N
        redk = plot_scaled_residual(A_hat_matrix(N, h, epsilon), f_hat_N(N, h, epsilon), N, TOL, gauss_seidel_iteration_method)
        red[k] = redk

    print(red)

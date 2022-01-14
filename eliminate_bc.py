import numpy as np
import matplotlib.pyplot as plt

def A_matrix(N,h,epsilon):
    A = np.zeros((N-1,N-1))
    np.fill_diagonal(A, (2*epsilon + h)/h**2)
    A += np.diag(np.full(N-2,-epsilon/h**2),1)

    A += np.diag(np.full(N-2,(-epsilon-h)/h**2),-1)
    return A

def f_N(N):
    f = np.ones(N-1)
    return f


if __name__ == "__main__":
    N = 100
    h = 1/N
    epsilon = 0.1

    v = np.zeros(N+1)
    v[1:-1] = np.linalg.solve(A_matrix(N,h,epsilon),f_N(N))
    x = np.linspace(0,1,N+1)

    u = v+1-x

    u_ref = (np.exp(x/epsilon) - np.exp(1/epsilon))/(1 - np.exp(1/epsilon))

    plt.plot(x,u, label="solution")
    plt.plot(x,u_ref, label="reference")
    plt.legend()
    plt.show()
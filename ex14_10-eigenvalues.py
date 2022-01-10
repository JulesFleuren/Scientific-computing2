import numpy as np
from matplotlib import pyplot as plt
# from main import *

def A_matrix(N,h, epsilon):
    dim = (N+1)**2
    A = np.zeros((dim,dim))

    ## TODO: check orderning, top left corner is 0,0, top left, bottom right corner is (N+1)**2,(N+1)**2

    ## Diffusion term (ddu/dxx)
    
    # Diagonal u(x,y) = 2
    np.fill_diagonal(A, 2/h**2)

    # Off-diagonal (left) u(x - 1,y) = -1
    np.fill_diagonal(A[:,1:], -1/h**2)

    # Off-diagonal (right) u(x + 1,y) = -1
    np.fill_diagonal(A[1:,:], -1/h**2)

    # Unset off-diagonals at boundary (redundant since executed every convetion term)

    # for k in range(1, N+1):
    #     A[k*(N+1)-1,k*(N+1)] = 0
    #     A[k*(N+1),k*(N+1)-1] = 0


    ## Convection term (du/dx)
    # Diagonal u(x,y) = h
    A += np.diagflat([1/h]*dim)

    # Off-diagonal (left) u(x - 1,y) = -h
    A += np.diagflat([-1/h]*(dim - 1), -1)

    # Unset off-diagonals at boundary
    for k in range(1, N+1):
        A[k*(N+1)-1,k*(N+1)] = 0
        A[k*(N+1),k*(N+1)-1] = 0


    ## Diffusion term (ddu/dyy)
    
    # Diagonal u(x,y) = 2
    A += np.diagflat([2]*dim)

    # Off-diagonal (bottom) u(x,y - 1) = -1
    A += np.diagflat([-1]*(dim - (N+1)), k=-(N+1))

    # Off-diagonal (top) u(x,y + 1) = -1
    A += np.diagflat([-1]*(dim - (N+1)), k=(N+1))

    return A

def f_N(N):
    # TODO: boundary conditions
    f = np.ones((N+1)**2)
    # f[0] = 1
    return f

def jacobi_eigenvalues(N, h, epsilon):
    A = A_matrix(N,h, epsilon)
    Dinv  = np.diag(1/np.diag(A))
    Bjac = np.identity((N+1)**2) - np.matmul(Dinv, A)
    # print(Bjac)
    w, v = np.linalg.eig(Bjac)
    return w


N=2
h=1/N
epsilon = 0.5
# print(A_matrix(N,h,epsilon))
w = jacobi_eigenvalues(N, h, epsilon)

plt.scatter(np.arange(len(w)), np.sort(w)[::-1])

plt.show()

print(w)
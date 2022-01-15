import numpy as np
from matplotlib import pyplot as plt
from main import *

def jacobi_eigenvalues(N, h, epsilon):
    A = A_matrix(N,h,epsilon)
    Dinv  = np.diag(1/np.diag(A))
    Bjac = np.identity(N+1) - np.matmul(Dinv, A)
    # print(Bjac)
    w, v = np.linalg.eig(Bjac)
    return w, v


N=10
h=1/N
epsilon = 0.5
w, v = jacobi_eigenvalues(N, h, epsilon)

print(w,v)

theta = np.linspace(0, 2*np.pi, 1000)
ellipsis = (np.sqrt(w[None,:]) * v) @ [np.sin(theta), np.cos(theta)]
plt.plot(ellipsis[0,:], ellipsis[1,:])
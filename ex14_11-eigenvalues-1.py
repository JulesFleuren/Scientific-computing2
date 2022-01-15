import numpy as np
from matplotlib import pyplot as plt
from ex14_main import *

def jacobi_eigenvalues(N, h):
    A = A_matrix(N,h)
    # Dinv  = np.diag(1/np.diag(A))
    # Bjac = np.identity(N-1) - np.matmul(Dinv, A)
    # print(Bjac)
    w, v = np.linalg.eig(A)
    return w

# Plot the eigenvalues given epsilon for various values of h
Ns = [2**n for n in range(6, 2, -1)]
for N in Ns:
    h = 1/N
    w = jacobi_eigenvalues(N, h)
    w = np.sort(w)[::-1]

    plt.scatter(np.arange(len(w)), np.sort(w)[::-1], label=r"$N={N}$".format(N=N))
    
plt.subplots_adjust(bottom=0.4)
plt.legend(bbox_to_anchor=(0,-0.8), loc="lower left", mode="expand", borderaxespad=0)
plt.title(r"Eigenvalues of the matrix $A$")
plt.show()
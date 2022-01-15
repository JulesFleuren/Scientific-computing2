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
    min_real = min(w.real)
    max_real = max(w.real)
    center_real = (min_real + max_real)/2
    radius_real = round(max_real - center_real, 2)

    min_imag = min(w.imag)
    max_imag = max(w.imag)
    center_imag = (min_imag + max_imag)/2
    radius_imag = round(max_imag - center_imag, 2)

    plt.scatter(w.real, w.imag, label=r"$N={N}, a={radius_real}, b={radius_imag}, O=({center_real},0i)$".format(N=N, radius_real=radius_real, radius_imag=radius_imag, center_real=round(center_real,2)))
plt.subplots_adjust(bottom=0.4)
plt.legend(bbox_to_anchor=(0,-0.8), loc="lower left", mode="expand", borderaxespad=0)
plt.title(r"Eigenvalues of the matrix $A$")
plt.show()
import numpy as np
from matplotlib import pyplot as plt
from main import *

def jacobi_eigenvalues(N, h, epsilon):
    A = A_hat_matrix(N,h,epsilon)
    # Dinv  = np.diag(1/np.diag(A))
    # Bjac = np.identity(N-1) - np.matmul(Dinv, A)
    # print(Bjac)
    w, v = np.linalg.eig(A)
    return w

def plotPerEpsilon(epsilon):
    # Plot the eigenvalues given epsilon for various values of h
    Ns = [2**n for n in range(3,10)]
    for N in Ns:
        h = 1/N
        w = jacobi_eigenvalues(N, h, epsilon)
        w = np.sort(w)[::-1]
        min_real = min(w.real)
        max_real = max(w.real)
        center_real = (min_real + max_real)/2
        radius_real = round(max_real - center_real, 2)

        min_imag = min(w.imag)
        max_imag = max(w.imag)
        center_imag = (min_imag + max_imag)/2
        radius_imag = round(max_imag - center_imag, 2)

        # origin = complex(round(center_real, 1), round(center_imag, 1))

        # print(radius_real, radius_imag, complex(center_real, center_imag))

        plt.scatter(w.real, w.imag, label=r"$N={N}, a={radius_real}, b={radius_imag}, O=({center_real},0i)$".format(N=N, radius_real=radius_real, radius_imag=radius_imag, center_real=round(center_real,2)))
    plt.subplots_adjust(bottom=0.4)
    plt.legend(bbox_to_anchor=(0,-0.8), loc="lower left", mode="expand", borderaxespad=0)
    plt.title(r"Eigenvalues of the matrix A for $\epsilon={epsilon:.0e}$".format(epsilon=epsilon))
    plt.show()

def plotPerN(N):
    # Plot the eigenvalues given N for various values of epsilon
    plt.figure()
    # epsilons = [10**n for n in range(-7,-1)]
    epsilons = [0.5]
    for epsilon in epsilons:
        h = 1/N
        w = jacobi_eigenvalues(N, h, epsilon)
        # w = np.sort(w)[::-1]
        min_real = min(w.real)
        max_real = max(w.real)
        center_real = (min_real + max_real)/2
        radius_real = round(max_real - center_real, 2)

        min_imag = min(w.imag)
        max_imag = max(w.imag)
        center_imag = (min_imag + max_imag)/2
        radius_imag = round(max_imag - center_imag, 2)

        
        # origin = complex(round(center_real, 1), round(center_imag, 1))

        # print(radius_real, radius_imag, complex(center_real, center_imag))

        plt.scatter(w.real, w.imag, label=r"$\epsilon={epsilon:.0e}, a={radius_real}, b={radius_imag}, O=({center_real},0i)$".format(epsilon=epsilon, radius_real=radius_real, radius_imag=radius_imag, center_real=round(center_real,2)))
        # plt.plot(np.arange(len(w)), np.sort(w)[::-1], label=f"epsilon={epsilon}")
    plt.subplots_adjust(bottom=0.4)
    plt.legend(bbox_to_anchor=(0,-0.8), loc="lower left", mode="expand", borderaxespad=0)
    plt.title(f"Eigenvalues of the matrix A for N={N}")

for N in [2**n for n in range(3,10)]:
    plotPerN(N)

# for epsilon in [10**n for n in range(-7,-1)]:
    # plotPerEpsilon(epsilon)

plt.show()
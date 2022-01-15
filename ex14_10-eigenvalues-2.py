import numpy as np
from matplotlib import pyplot as plt
from ex14_main import *

def jacobi_eigenvalues(N, h):
    A = A_matrix(N,h)
    # Dinv  = np.diag(1/np.diag(A))
    # DinvA = np.matmul(Dinv, A)

    DinvA = h**2/4*A
    Bjac = np.identity((N-1)**2) - DinvA
    # Compute eigenvalues of Bjac
    w, v = np.linalg.eig(Bjac)
    return w


# def plotPerEpsilon(epsilon):
# Plot the eigenvalues given epsilon for various values of h
Ns = [2**n for n in range(6, 2, -1)]
for N in Ns:
    h = 1/N
    w = jacobi_eigenvalues(N, h)
    w = np.sort(w)[::-1]

    plt.scatter(np.arange(len(w)), np.sort(w)[::-1], label=r"$N={N}$".format(N=N))

plt.legend()
plt.title(r"Eigenvalues of the matrix $A$")
plt.show()

# def plotPerN(N):
#     # Plot the eigenvalues given N for various values of epsilon
#     plt.figure()
#     epsilons = [10**n for n in range(0, -6, -1)]
#     for epsilon in epsilons:
#         h = 1/N
#         w = jacobi_eigenvalues(N, h, epsilon)
#         w = np.sort(w)[::-1]
#         # min_real = min(w.real)
#         # max_real = max(w.real)
#         # center_real = (min_real + max_real)/2
#         # radius_real = round(max_real - center_real, 2)

#         # min_imag = min(w.imag)
#         # max_imag = max(w.imag)
#         # center_imag = (min_imag + max_imag)/2
#         # radius_imag = round(max_imag - center_imag, 2)

        
#         # origin = complex(round(center_real, 1), round(center_imag, 1))

#         # print(radius_real, radius_imag, complex(center_real, center_imag))

#         plt.scatter(w.real, w.imag, label=r"$\epsilon={epsilon:.0e}$".format(epsilon=epsilon))
#         # plt.scatter(w.real, w.imag, label=r"$\epsilon={epsilon:.0e}, a={radius_real}, b={radius_imag}, O=({center_real},0i)$".format(epsilon=epsilon, radius_real=radius_real, radius_imag=radius_imag, center_real=round(center_real,2)))
#         # plt.plot(np.arange(len(w)), np.sort(w)[::-1], label=f"epsilon={epsilon}")
#     # plt.subplots_adjust(bottom=0.4)
#     plt.legend()
#     plt.title(f"Eigenvalues of the Jacobi iteration matrix for N={N}")

# for N in [2**n for n in range(3,10)]:
#     plotPerN(N)

# for epsilon in [10**n for n in range(-7,-6)]:
    # plotPerEpsilon(epsilon)

plt.show()
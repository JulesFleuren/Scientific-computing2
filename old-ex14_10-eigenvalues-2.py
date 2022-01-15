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

# Plot the eigenvalues for various values of N
Ns = [2**n for n in range(6, 2, -1)]

plt.figure()
for N in Ns:
    h = 1/N
    w = jacobi_eigenvalues(N, h)
    w = np.sort(w)[::-1]

    plt.scatter(np.arange(len(w)), np.sort(w)[::-1], label=r"$N={N}$".format(N=N))
# plt.subplots_adjust(bottom=0.4)
plt.legend()
plt.title(f"Eigenvalues of the 2D problem for various values of N")

plt.show()

# if __name__ == "__main__":
#     N=10
#     h=1/N
#     epsilon = 0.5

#     print(A_matrix(N,h,epsilon))
#     # print(Add_Dirichet_Boundary(A_matrix(N,h,epsilon)))
#     # print(f_N(2))
#     print(f_N(N))

#     w = jacobi_eigenvalues(N, h, epsilon)

#     plt.scatter(np.arange(len(w)), np.sort(w)[::-1])

#     plt.show()


#     U = np.linalg.solve(A_matrix(N,h,epsilon), f_N(N))
#     matrix_u = np.reshape(U, (N-1,N-1))
#     matrix_u = Add_Dirichet_Boundary(matrix_u)

#     # plt.(matrix_u)
#     # 3D plot surface plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X = np.arange(0, N+1, 1)
#     Y = np.arange(0, N+1, 1)
#     X, Y = np.meshgrid(X, Y)
#     ax.plot_surface(X, Y, matrix_u, cmap='viridis')
#     plt.show()

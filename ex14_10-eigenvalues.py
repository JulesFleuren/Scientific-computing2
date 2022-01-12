import numpy as np
from matplotlib import pyplot as plt
from ex14_main import *

def jacobi_eigenvalues(N, h, epsilon):
    A = A_matrix(N,h, epsilon)
    Dinv  = np.diag(1/np.diag(A))
    Bjac = np.identity((N-1)**2) - np.matmul(Dinv, A)
    # print(Bjac)
    w, v = np.linalg.eig(Bjac)
    return w


N=10
h=1/N
epsilon = 0.5

print(A_matrix(N,h,epsilon))
# print(Add_Dirichet_Boundary(A_matrix(N,h,epsilon)))
# print(f_N(2))
print(f_N(N))

w = jacobi_eigenvalues(N, h, epsilon)

plt.scatter(np.arange(len(w)), np.sort(w)[::-1])

plt.show()

print(w)



U = np.linalg.solve(A_matrix(N,h,epsilon), f_N(N))
matrix_u = np.reshape(U, (N-1,N-1))
matrix_u = Add_Dirichet_Boundary(matrix_u)

# plt.(matrix_u)
# 3D plot surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(0, N+1, 1)
Y = np.arange(0, N+1, 1)
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, matrix_u, cmap='viridis')
plt.show()
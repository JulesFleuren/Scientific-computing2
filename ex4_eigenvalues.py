import numpy as np
from matplotlib import pyplot as plt
from main import A_matrix


N=20
h=1/N
epsilon = 1
w1, v = np.linalg.eig(A_matrix(N,h,epsilon))
epsilon = 0.5
w2, v = np.linalg.eig(A_matrix(N,h,epsilon))
epsilon = 0.1
w3, v = np.linalg.eig(A_matrix(N,h,epsilon))
print(np.iscomplex(w1).any())
print(np.iscomplex(w2).any())
print(np.iscomplex(w3).any())

# plt.plot(np.arange(len(w)),N**2+ N**2*(1-np.cos(np.linspace(0,np.pi,len(w)))), c="orange")
# plt.plot(np.arange(len(w)),2*N**2*(1-np.cos(np.linspace(0,np.pi,len(w)))), c="orange")
plt.scatter(np.arange(len(w1)), np.sort(w1), label=r"$\epsilon = 1$")
plt.scatter(np.arange(len(w2)), np.sort(w2), label=r"$\epsilon = 0.5$")
plt.scatter(np.arange(len(w3)), np.sort(w3), label=r"$\epsilon = 0.1$")
plt.legend()
plt.show()

import numpy as np
from scipy.sparse import dia_matrix, identity, csr_matrix
from scipy.sparse.linalg import inv
from main import *

N=100
h=1/N
epsilon = 0.5
A = A_matrix(N,h,epsilon)
D = dia_matrix((1./np.diag(A), [0]), shape=(N, N))
# D = np.diag(A)

# Bjac = identity(N) - np.multiply(A.T, D)

Bjac = identity(N) - D*A
print(D.shape)
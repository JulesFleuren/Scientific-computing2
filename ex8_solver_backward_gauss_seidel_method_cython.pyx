import numpy as np
cimport numpy as np
from cpython cimport bool
np.import_array()

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing

cdef double norm(np.ndarray[DTYPE_t, ndim=1] x):
    cdef int i
    cdef DTYPE_t s = 0
    for i in range(len(x)):
        s += x[i]*x[i]
    return np.sqrt(s)

def backward_gauss_seidel_iteration_method(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=1] f, double TOL, bool tridiagonal=False):
    cdef int len_f = f.shape[0]

    assert A.shape[0] == len_f
    assert A.shape[1] == len_f

    cpdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros(len_f, dtype=DTYPE)
    cdef DTYPE_t v
    cdef DTYPE_t norm_r_squared
    
    cdef int i
    cdef int j
    cdef DTYPE_t s

    cdef list sr = []
    cdef double norm_f = norm(f)
    cdef double res = 2*TOL

    if not tridiagonal:
        while res > TOL:

            for i in range(len_f-1, -1, -1): # loop backwards
                s = f[i]
                for j in range(len_f):
                    s -= A[i,j] * u[j]
                u[i] += s/A[i,i]

            # res = np.linalg.norm(f - np.matmul(A, u))/np.linalg.norm(f)
            norm_r_squared = 0
            for i in range(len_f):
                v = f[i]
                for j in range(len_f):
                    v -= A[i,j]*u[j]
                norm_r_squared += v*v
                
            res = np.sqrt(norm_r_squared)/norm_f

            sr.append(res)
    
    elif tridiagonal:
        while res > TOL:

            for i in range(len_f-1, -1, -1):
                s = f[i]
                
                s -= A[i,i] * u[i]
                if i > 0:
                    s -= A[i,i-1] * u[i-1]
                if i < len_f-1:
                    s -= A[i,i+1] * u[i+1]
                
                u[i] += s/A[i,i]

            norm_r_squared = 0
            for i in range(len_f):
                v = f[i]

                v -= A[i,i] * u[i]
                if i > 0:
                    v -= A[i,i-1] * u[i-1]
                if i < len_f-1:
                    v -= A[i,i+1] * u[i+1]

                norm_r_squared += v*v
                
            res = np.sqrt(norm_r_squared)/norm_f

            sr.append(res)

    return u, sr, len(sr)
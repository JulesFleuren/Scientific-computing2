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

def jacobi_iteration_method(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=1] f, const double TOL, bool tridiagonal=False):
    cdef int len_f = f.shape[0]

    assert A.shape[0] == len_f
    assert A.shape[1] == len_f

    cdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros(len_f, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] r = np.zeros(len_f, dtype=DTYPE)
    cdef DTYPE_t norm_r_squared = 0
    
    cdef int i
    cdef int j
    cdef DTYPE_t s

    cdef list sr = []
    cdef double norm_f = norm(f)
    cdef double res = 2*TOL
        
    # Initial residual
    r = f.copy()
    for i in range(len_f):
        for j in range(len_f):
            r[i] -= A[i,j] * u[j]

    if not tridiagonal:
        while res > TOL:
            for i in range(len_f):
                u[i] += r[i]/A[i,i]
            res = 0
            r = f.copy()
            for i in range(len_f):
                # u[i] = u[i] + (f[i] - np.sum(A[i,:]*u))/A[i,i]
                for j in range(len_f):
                    r[i] -= A[i,j] * u[j]

            # np.linalg.norm(f - np.matmul(A, u))/np.linalg.norm(f)
            #res = norm(f - np.matmul(A, u))/norm_f
            res = norm(r)/norm_f

            sr.append(res)
    
    elif tridiagonal:
        while res > TOL:
            for i in range(len_f):
                u[i] += r[i]/A[i,i]
            res = 0
            r = f.copy()
            for i in range(len_f):
                r[i] -= A[i,i] * u[i]
                if i > 0:
                    r[i] -= A[i,i-1] * u[i-1]
                if i < len_f-1:
                    r[i] -= A[i,i+1] * u[i+1]

            res = norm(r)/norm_f

            sr.append(res)
        
    return u, sr, len(sr)

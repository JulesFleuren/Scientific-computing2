import numpy as np
cimport numpy as np
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
        s += x[i]**2
    return np.sqrt(s)

def gauss_seidel_iteration_method(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=1] f, double TOL):

    cdef int len_f = f.shape[0]

    assert A.shape[0] == len_f
    assert A.shape[1] == len_f

    cdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros(len_f, dtype=DTYPE)
    
    cdef int i
    cdef int j
    cdef DTYPE_t s

    cdef list sr = []

    cdef double normf = norm(f)
    cdef double TOL_f = TOL*normf
    cdef double res = 2*TOL

    
    while res > TOL_f:

        for i in range(len_f):
            # u[i] = u[i] + (f[i] - np.sum(A[i,:]*u))/A[i,i]
            s = f[i]
            for j in range(len_f):
                s -= A[i,j] * u[j]
            u[i] += s/A[i,i]

        res = norm(f - np.matmul(A, u))
        sr.append(res)
    
    return u, sr, len(sr)

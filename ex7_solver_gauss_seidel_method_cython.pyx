import numpy as np
cimport numpy as np
np.import_array()

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing

def gauss_seidel_iteration_method(np.ndarray[DTYPE_t, ndim=2] A, np.ndarray[DTYPE_t, ndim=1] f, double TOL):

    cdef int len_f = f.shape[0]

    assert A.shape[0] == len_f
    assert A.shape[1] == len_f

    cpdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros(len_f, dtype=DTYPE)
    

    cdef int k = 0
    cdef int i
    cdef int j
    cdef DTYPE_t s

    sr = []

    cdef double res = 2*TOL
    while res > TOL:

        for i in range(len_f):
            # u[i] = u[i] + (f[i] - np.sum(A[i,:]*u))/A[i,i]
            s = 0
            for j in range(len_f):
                s += A[i,j] * u[j]
            u[i] = u[i] + (f[i] - s)/A[i,i]

        res = np.linalg.norm(f - np.matmul(A, u))/np.linalg.norm(f)
        sr.append(res)
        k += 1
    return u, sr, k

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

    cdef extern from "math.h":
        double sqrt(double x)

    cdef int len_f = f.shape[0]

    assert A.shape[0] == len_f
    assert A.shape[1] == len_f

    cpdef np.ndarray[DTYPE_t, ndim=1] u = np.zeros(len_f, dtype=DTYPE)
    cpdef np.ndarray[DTYPE_t, ndim=1] v = np.zeros(len_f, dtype=DTYPE)
    
    cdef int i
    cdef int j
    cdef DTYPE_t s

    cdef list sr = []
    cdef double norm_f = norm(f)
    cdef double res = 2*TOL

    
    while res > TOL:

        for i in range(len_f):
            # u[i] = u[i] + (f[i] - np.sum(A[i,:]*u))/A[i,i]
            s = f[i]
            for j in range(len_f):
                s -= A[i,j] * u[j]
            u[i] += s/A[i,i]

        # res = np.linalg.norm(f - np.matmul(A, u))/np.linalg.norm(f)
        res = 0
        for i in range(len_f):
            v[i] = f[i]
            for j in range(len_f):
                v[i] -= A[i,j]*u[j]
            
        res = norm(v)/norm_f

        sr.append(res)
    
    return u, sr, len(sr)

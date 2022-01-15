import numpy as np
import scipy.sparse.linalg as spla


from ex14_main import *


def GMRES_method_non_full(A, f, u_0, TOL, iterations=10):
    
    len_f = f.shape[0]
    maxiter = len_f+1

    r_0 = f - np.matmul(A,u_0)

    v_0 = r_0/ np.linalg.norm(r_0)
    V = np.zeros((len_f, iterations+1))
    V[:,0] = v_0        # matrix met alle v's als kolommen. v vectors zijn basis voor de Krylov space, die gevonden worden met Arnoldi's method

    beta = np.zeros(iterations+1)        # beta e_1 vector 
    beta[0] = np.linalg.norm(r_0)

    j=0
    res = TOL+1
    sr = np.zeros(iterations+1)

    H = np.zeros((iterations+1,iterations))       # Hessenberg matrix
    
    while res > TOL and j < maxiter and j < iterations:
        
        v_new = np.matmul(A,V[:,j])
        for i in range(j+1):
            H[i,j] = np.dot(v_new, V[:,i])
            v_new = v_new - H[i,j]*V[:,i]
        H[j+1,j] = np.linalg.norm(v_new)
        
        if H[j+1,j] != 0:
            v_new = v_new/H[j+1,j]
        V[:,j+1] = v_new
        
        y_new = np.linalg.lstsq(H[:j+2,:j+1], beta[:j+2], rcond=None)[0]
        u_new = u_0 + np.matmul(V[:,:j+1],y_new)

        # print(f"iteration: {j}, u_new: {u_new}")

        
        r_new = f-np.matmul(A,u_new)
        res = np.linalg.norm(r_new)/np.linalg.norm(f)
        sr[j] = res
        
        if H[j+1,j] == 0:
            j+=1
            break
        
        j+=1

    return u_new, sr[:j], j

def repeated_GMRES_method(A, f, u_0, TOL, iterations):

    # preconditioning
    D_inv = np.diag(1/np.diag(A))
    # D_inv = np.diag(np.ones(f.shape[0]))
    A_new = np.matmul(D_inv, A)
    f_new = np.matmul(D_inv, f)

    j=0
    res = TOL+1
    sr = []
    u_old = u_0
    while res > TOL:
        u_new, sr_iter, j_iter = GMRES_method_non_full(A_new, f_new, u_old, TOL, iterations)
        sr += list(sr_iter)
        res = sr_iter[-1]
        j += j_iter
        u_old = u_new
    return u_new, sr, j

if __name__ == "__main__":
    N=13
    h=1/N
    epsilon = 0.5
    iterations = 13

    print(repeated_GMRES_method(A_matrix(N, h, epsilon), f_N(N), f_N(N), 1e-6, iterations))
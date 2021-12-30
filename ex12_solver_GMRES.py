import numpy as np
import scipy.sparse.linalg as spla



def GMRES_method(A, f, u_0, TOL):
    
    len_f = f.shape[0]
    maxiter = len_f+1

    # preconditioning
    D_inv = np.diag(1/np.diag(A))
    # D_inv = np.diag(np.ones(len_f))
    A_new = np.matmul(D_inv, A)
    f_new = np.matmul(D_inv, f)

    r_0 = f_new - np.matmul(A_new,u_0)

    v_0 = r_0/ np.linalg.norm(r_0)
    V = np.zeros((len_f, maxiter))
    V[:,0] = v_0        # matrix met alle v's als kolommen. v vectors zijn basis voor de Krylov space, die gevonden worden met Arnoldi's method

    beta = np.zeros(maxiter)        # beta e_1 vector 
    beta[0] = np.linalg.norm(r_0)

    j=0
    res = TOL+1
    sr = np.zeros(maxiter)

    H = np.zeros((maxiter+1,maxiter))       # Hessenberg matrix
    
    while res > TOL and j < maxiter-1:
        
        v_new = np.matmul(A_new,V[:,j])
        for i in range(j+1):
            H[i,j] = np.dot(v_new, V[:,i])
            v_new = v_new - H[i,j]*V[:,i]
        H[j+1,j] = np.linalg.norm(v_new)
        
        if H[j+1,j] != 0:
            v_new = v_new/H[j+1,j]
        V[:,j+1] = v_new
        
        y_new = np.linalg.lstsq(H[:j+2,:j+1], beta[:j+2])[0]
        u_new = u_0 + np.matmul(V[:,:j+1],y_new)

        # print(f"iteration: {j}, u_new: {u_new}")

        
        r_new = f_new-np.matmul(A_new,u_new)
        res = np.linalg.norm(r_new)/np.linalg.norm(f_new)
        sr[j] = res
        
        if H[j+1,j] == 0:
            j+=1
            break
        
        j+=1
        
    if j == maxiter:
        raise RuntimeError("max iterations reached without converging")


    return u_new, sr[:j], j

if __name__ == "__main__":
    M = np.array([[1,1],[3,-4]])
    b = np.array([3,2])
    u_0 = np.array([1,2])
    print(np.linalg.solve(M,b))
    print(GMRES_method(M,b,u_0,1e-8))
    print(spla.gmres(M,b,u_0))
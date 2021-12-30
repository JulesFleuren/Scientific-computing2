from main import A_matrix, f_N
import numpy as np
import scipy.sparse.linalg as spla

N=10
h=1/N
epsilon = 0.5

A = A_matrix(N,h,epsilon)
f = f_N(N)

u_0 = np.zeros(f.shape)

def GMRES_method(A, f, u_0, TOL, maxiter=1e10):
    len_f = f.shape[0]

    r_0 = f - np.matmul(A,u_0)

    v_1 = r_0/ np.linalg.norm(r_0)
    V = np.zeros((len_f, maxiter))
    V[:,0] = v_1        # matrix met alle v's als kolommen. v vectors zijn basis voor de Krylov space, die gevonden worden met Arnoldi's method

    beta = np.zeros(maxiter)        # beta e_1 vector 
    beta[0] = np.linalg.norm(r_0)

    j=0
    res = TOL+1

    H = np.zeros((maxiter+1,maxiter))       # Hessenberg matrix
    
    while res > TOL and j < maxiter:
        
        v_new = np.matmul(A,V[:,j])
        for i in range(j):
            H[i,j] = np.dot(v_new, V[:,i])
            v_new = v_new - H[i,j]*V[:,i]
        H[j+1,j] = np.linalg.norm(v_new)
        
        if H[j+1,j] != 0:
            v_new = v_new/H[j+1,j]
        V[:,j+1] = v_new
        
        y_new = np.linalg.lstsq(H[:j+1,:j], beta[:j+1])[0]
        u_new = u_0 + np.matmul(V[:,:j],y_new)

        # print(f"iteration: {j}, u_new: {u_new}")

        if H[j+1,j] == 0:
            print("H = 0")
            break
        
        r_new = f-np.matmul(A,u_new)
        res = np.linalg.norm(r_new)/np.linalg.norm(f)
        
        j+=1
    return u_new, j

if __name__ == "__main__":
    M = np.array([[1,1],[3,-4]])
    b = np.array([3,2])
    u_0 = np.array([1,2])
    print(np.linalg.solve(M,b))
    print(GMRES_method(M,b,u_0,1e-8,100))
    print(spla.gmres(M,b,u_0))
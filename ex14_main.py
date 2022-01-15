import numpy as np

def A_matrix(N,h):
    dim = (N-1)**2
    A = np.zeros((dim,dim))

    ## TODO: check orderning, top left corner is 0,0, top left, bottom right corner is (N+1)**2,(N+1)**2

    ## Diffusion term (ddu/dxx)
    
    # Diagonal u(x,y) = 2
    np.fill_diagonal(A, 2/h**2)

    # Off-diagonal (left) u(x - 1,y) = -1
    np.fill_diagonal(A[:,1:], -1/h**2)

    # Off-diagonal (right) u(x + 1,y) = -1
    np.fill_diagonal(A[1:,:], -1/h**2)

    ## Convection term (du/dx)
    # Diagonal u(x,y) = h
    A += np.diagflat([1/h]*dim)

    # Off-diagonal (left) u(x - 1,y) = -h
    A += np.diagflat([-1/h]*(dim - 1), -1)

    # Unset off-diagonals at boundary
    for k in range(1, N-1):
        A[k*(N-1)-1,k*(N-1)] = 0
        A[k*(N-1),k*(N-1)-1] = 0


    ## Diffusion term (ddu/dyy)
    
    # Diagonal u(x,y) = 2
    A += np.diagflat([2/h**2]*dim)

    # Off-diagonal (bottom) u(x,y - 1) = -1
    A += np.diagflat([-1/h**2]*(dim - (N-1)), k=-(N-1))

    # Off-diagonal (top) u(x,y + 1) = -1
    A += np.diagflat([-1/h**2]*(dim - (N-1)), k=(N-1))

    return A

def Add_Dirichet_Boundary(M):
    M2 = np.zeros((M.shape[0]+2, M.shape[1]+2))
    M2[1:-1,1:-1] = M
    return M2

def f_N(N):
    # TODO: check corner boundary conditions
    f = np.ones((N-1)**2)
    
    # # RHS for x = 0
    # f[0:N+1] += 1
    # # RHS for x = N+1
    # f[N*(N+1):] += 1

    # for k in range(0, N+1):
    #     # RHS for y = 0
    #     f[k*(N+1)] += 1

    #     # RHS for y = N+1
    #     f[k*(N+1)+N] += 1

    return f
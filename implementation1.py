import numpy as np
import matplotlib.pyplot as plt


epsilon = 0.1
def direct_solve(N,h,epsilon):
    A = A_matrix(N,h,epsilon)

    f = np.zeros(N+1)
    f[0] = 1

    u = np.linalg.solve(A,f)
    x = np.linspace(0,1,N+1)
    return x, u

def A_matrix(N,h,epsilon):
    A = np.zeros((N+1,N+1))
    np.fill_diagonal(A, (2*epsilon + h)/h**2)
    A[0,0] = 1
    A[-1,-1] = 1
    A += np.diag(np.full(N,-epsilon/h**2),1)
    A[0,1] = 0

    A += np.diag(np.full(N,(-epsilon-h)/h**2),-1)
    A[-1,-2] = 0
    return A




# # ex. 1
# N=4
# h=1/N
# epsilon = np.logspace(0,-4,8)
# x_ref = np.linspace(0,1,50)
# X_ref, Epsilon = np.meshgrid(x_ref,epsilon)
# epsilon = 0.5
# U_ref = (np.exp(X_ref/Epsilon) - np.exp(1/Epsilon))/(1 - np.exp(1/Epsilon))

# plt.plot(*direct_solve(N,h,epsilon), label="numerical solution")
# plt.plot(X_ref.T, U_ref.T, label="reference solution")

# plt.legend()

# table of errors
epsilon = 0.5
print("epsilon", "N", "error", sep = '\t')
for epsilon in [0.5, 0.1, 0.01]:
    for N in [2**n for n in range(4,12)]:
        x, u = direct_solve(N,1/N,epsilon)
        u_ref =  (np.exp(x/epsilon) - np.exp(1/epsilon))/(1 - np.exp(1/epsilon))
        max_difference = np.max(np.abs(u-u_ref))
        print(epsilon, N, max_difference, sep="\t")

# # Richardson error estimate
# N = 128
# h = 1/N
# epsilon = 0.5
# _, u_1 = direct_solve(N,h,epsilon)
# _, u_2 = direct_solve(N//2,2*h,epsilon)
# _, u_4 = direct_solve(N//4,4*h,epsilon)
# i=N//2
# p = np.log2((u_2[i//2]-u_4[i//4])/(u_1[i]-u_2[i//2]))
# print("order p of accuracy of solution:", p)

# eigenvalues, eigenvectors of A
N=100
h=1/N
epsilon = 0.5
w, v = np.linalg.eig(A_matrix(N,h,epsilon))
print(w)

plt.scatter(np.arange(len(w)), np.sort(w))

# plt.scatter(np.arange(len(w)), v[:,0])
# plt.scatter(np.arange(len(w)), v[:,1])
# plt.scatter(np.arange(len(w)), v[:,2])
# plt.scatter(np.arange(len(w)), v[:,3])
# plt.scatter(np.arange(len(w)), v[:,4])
# plt.scatter(np.arange(len(w)), v[:,5])
# plt.scatter(np.arange(len(w)), v[:,6])
# plt.scatter(np.arange(len(w)), v[:,7])
# plt.scatter(np.arange(len(w)), v[:,8])
# plt.scatter(np.arange(len(w)), v[:,9])



plt.show()
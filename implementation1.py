import numpy as np
import matplotlib.pyplot as plt


epsilon = 0.5
def direct_solve(N,h,epsilon):
    A = np.zeros((N+1,N+1))
    np.fill_diagonal(A, (2*epsilon + h)/h**2)
    A[0,0] = 1
    A[-1,-1] = 1
    A += np.diag(np.full(N,-epsilon/h**2),1)
    A[0,1] = 0

    A += np.diag(np.full(N,(-epsilon-h)/h**2),-1)
    A[-1,-2] = 0

    f = np.zeros(N+1)
    f[0] = 1

    u = np.linalg.solve(A,f)
    x = np.linspace(0,1,N+1)
    return x, u

# # table of errors
# epsilon = 0.5
# print("epsilon", "N", "error", sep = '\t')
# for epsilon in [0.5, 0.1, 0.01]:
#     for N in [2**n for n in range(4,12)]:
#         x, u = direct_solve(N,1/N,epsilon)
#         u_ref =  (np.exp(x/epsilon) - np.exp(1/epsilon))/(1 - np.exp(1/epsilon))
#         max_difference = np.max(np.abs(u-u_ref))
#         print(epsilon, N, max_difference, sep="\t")

# Richardson error estimate
N = 128
h = 1/N
_, u_1 = direct_solve(N,h,epsilon)
_, u_2 = direct_solve(N//2,2*h,epsilon)
_, u_4 = direct_solve(N//4,4*h,epsilon)
i=N//2
p = np.log2((u_2[i//2]-u_4[i//4])/(u_1[i]-u_2[i//2]))
print("order p of accuracy of solution:", p)

# x_ref = np.linspace(0,1,50)
# u_ref = (np.exp(x_ref/epsilon) - np.exp(1/epsilon))/(1 - np.exp(1/epsilon))

# plt.plot(*direct_solve(), label="numerical solution")
# plt.plot(x_ref, u_ref, label="reference solution")

# plt.legend()
# plt.show()
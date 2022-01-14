from main import *

print("epsilon", "N", "error", sep = '\t &')
for epsilon in [0.5, 0.1, 0.01]:
    for N in [2**n for n in range(4,12)]:
        x, u = direct_solve(N,1/N,epsilon)
        max_difference = np.max(np.abs(u - u_ex(x,epsilon)))
        print(N, f"{max_difference:.2e}", sep="\t& ", end="\t \\\\ \n")
        # print(epsilon, N, f"{max_difference:.2e}", sep="\t& ", end="\t \\\\ \n")

# Richardson error estimate
N = 128
h = 1/N
epsilon = 0.5
_, u_1 = direct_solve(N,h,epsilon)
_, u_2 = direct_solve(N//2,2*h,epsilon)
_, u_4 = direct_solve(N//4,4*h,epsilon)
i=N//2
p = np.log2((u_2[i//2]-u_4[i//4])/(u_1[i]-u_2[i//2]))
print("order p of accuracy of solution:", p)
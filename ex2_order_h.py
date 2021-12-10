from main import *

print("epsilon", "N", "error", sep = '\t')
for epsilon in [0.5, 0.1, 0.01]:
    for N in [2**n for n in range(4,12)]:
        x, u = direct_solve(N,1/N,epsilon)
        max_difference = np.max(np.abs(u - u_ex(x,epsilon)))
        print(epsilon, N, max_difference, sep="\t")

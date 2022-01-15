import numpy as np
import matplotlib.pyplot as plt
from main import A_hat_matrix, f_hat_N, plot_scaled_residual3
from ex14_main import A_matrix, f_N

import ex13_solver_repeated_GMRES as rgmres

fig, ax = plt.subplots()

N = 2**5
h = 1/N
epsilon = 0.5
TOL = 1e-6

for m in [2**n for n in range(1,8)]:
    h = 1/N
    u_0 = np.zeros(N-1)
    # redf = plot_scaled_residual3(rgmres.repeated_GMRES_method, ax, A_hat_matrix(N,h,epsilon), f_hat_N(N, h, epsilon), u_0, TOL, m)
    u_0 = np.zeros((N-1)**2)
    redf = plot_scaled_residual3(rgmres.repeated_GMRES_method, ax, A_matrix(N,h,epsilon), f_N(N), u_0, TOL, m)
    # print(N, *[f"{r:.5f}" for r in redf], sep = " & ", end = "\\\\\n")



ax.set_xlabel('Iteration')
ax.set_ylabel('Scaled Residual')
ax.set_yscale('log')
ax.set_ylim(1e-7, 1)
ax.legend()

plt.show()
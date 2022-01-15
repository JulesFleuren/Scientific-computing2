import numpy as np
import matplotlib.pyplot as plt
# from main import A_hat_matrix, f_hat_N, plot_scaled_residual2
from main import plot_scaled_residual2
from ex14_main import A_matrix, f_N
import ex6_solver_jacobi_method_cython as jac
import ex7_solver_gauss_seidel_method_cython as gs
import ex8_solver_backward_gauss_seidel_method_cython as bgs
import ex9_solver_symmetric_gauss_seidel_method_cython as sgs
import ex12_solver_GMRES as gmres

fig, ax = plt.subplots()

epsilon = 0.5
TOL = 1e-6
for N in [2**n for n in range(4,7)]:
    h = 1/N
    # u_0 = np.zeros(N-1)
    # redf = plot_scaled_residual2(gmres.GMRES_method, ax, A_hat_matrix(N,h,epsilon), f_hat_N(N, h, epsilon), u_0, TOL)
    u_0 = np.zeros((N-1)**2)
    redf = plot_scaled_residual2(gmres.GMRES_method, ax, A_matrix(N,h,epsilon), f_N(N), u_0, TOL)

    # print(N, *[f"{r:.5f}" for r in redf], sep = " & ", end = "\\\\\n")



# k = np.arange(2**8)
# ax.plot(k, 2.6867 * (0.999917)**k, label="theoretical bound N = 256")

ax.set_xlabel('Iteration')
ax.set_ylabel('Scaled Residual')
ax.set_yscale('log')
# ax.set_ylim(1e-4, 5)

ax.legend()

plt.show()

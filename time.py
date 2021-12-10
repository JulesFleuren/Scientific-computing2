import numpy as np
import matplotlib.pyplot as plt

import ex7_solver_gauss_seidel_method_cython as gs_cy
from timeit import timeit

from main import A_matrix, f_N

setup1 = """
from main import A_matrix, f_N
import ex7_solver_gauss_seidel_method_cython as gs_cy
import ex8_solver_backward_gauss_seidel_method_cython as bgs_cy
import ex7_solver_gauss_seidel_method as gs_py

N = 2**6
h = 1/N
epsilon = 0.1

A = A_matrix(N, h, epsilon)
f = f_N(N)
TOL = 1e-6
"""

ctime = timeit(stmt="gs_cy.gauss_seidel_iteration_method(A,f,TOL)", setup=setup1, number =13)
print(ctime)
# ctime2 = timeit(stmt="bgs_cy.backward_gauss_seidel_iteration_method(A,f,TOL)", setup=setup1, number =10)
# print("cython done")
ptime = timeit(stmt="gs_py.gauss_seidel_iteration_method(A,f,TOL)", setup=setup1, number =13)

# print(ctime, ctime2, ptime)
# print(ctime/ctime2, ptime/ctime)

print(ctime, ptime, ptime/ctime)

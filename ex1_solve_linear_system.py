import numpy as np
import matplotlib.pyplot as plt

from main import *


N=64
h=1/N
epsilon = np.logspace(0,-2,N)
# x_ref = np.linspace(0,1,256)
# X_ref, Epsilon = np.meshgrid(x_ref,epsilon)

# U_ref = u_ex(X_ref, Epsilon)

epsilons = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9]
# epsilons = [10**n for n in range(0, -6, -1)]
for epsilon in epsilons:
    x_ref = np.linspace(0,1,256)
    u_ref = u_ex(x_ref, epsilon)
    plt.plot(x_ref, u_ref, label=f"Numerical solution (N={N}, ε={epsilon})")


plt.plot(*direct_solve(N,h, 1), label=f"Reference solution (N={N}, ε={1})")
# labels = [f"Numerical solution ε={i}" for i in epsilon]
# plt.plot(X_ref.T, U_ref.T, label="Numerical Solution")

plt.legend()

plt.show()

# TA vragen:
# - Moeten we eerst h variëren en daarna epsilon?

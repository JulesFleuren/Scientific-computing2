import numpy as np
import matplotlib.pyplot as plt

from main import *


N=5
h=1/N
epsilon = np.logspace(0,-2,N)
x_ref = np.linspace(0,1,50)
X_ref, Epsilon = np.meshgrid(x_ref,epsilon)

U_ref = u_ex(X_ref, Epsilon)

plt.plot(*direct_solve(N,h, 1), label=f"Reference solution (N={N}, h={h}, ε={1})")


# labels = [f"Numerical solution ε={i}" for i in epsilon]
plt.plot(X_ref.T, U_ref.T, label="Numerical Solution")

plt.legend()

plt.show()
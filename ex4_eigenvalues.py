import numpy as np
from matplotlib import pyplot as plt
from main import *

N=10
h=1/N
epsilon = 0.5
w, v = np.linalg.eig(A_matrix(N,h,epsilon))
print(w)

plt.scatter(np.arange(len(w)), np.sort(w))

plt.show()
